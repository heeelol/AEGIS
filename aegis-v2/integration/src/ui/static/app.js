/**
 * AEGIS — Kitting Operator Frontend
 * ==================================
 * Count-driven kitting view:
 *   - Bin tiles (missing / not-in-BOM / incomplete / complete / overpick)
 *   - Hand-in-bin indicator + wrong-bin cross overlay
 *   - Kit box (placed / total, COMPLETED state)
 *   - Status bar (the single next-action instruction)
 *   - Full-screen corrective overlay (driven only by the 4 backend alert events)
 *
 * Data contract:
 *   /api/bins  -> [{id,label,in_bom,detected,current,target,hand,handedness}]
 *   /api/kit   -> {placed,total,batch:{done,target}}
 *   /api/alert -> {active,message}
 *   /api/layout-> {layers:[{layer,row_slots,bins:[{id,slot_start,span}]}]}
 */

const POLL_INTERVAL = 1000; // ms

// ── Bin status derivation ───────────────────────────
function binStatus(b) {
  if (!b.detected) return "missing";
  if (!b.in_bom || b.target <= 0) return "not_in_bom";
  if (b.current > b.target) return "overpick";
  if (b.current >= b.target) return "complete";
  return "incomplete";
}
function isWrongBin(b, status) {
  return b.hand && (status === "not_in_bom" || status === "complete");
}

// ── Main poll loop ──────────────────────────────────
async function poll() {
  try {
    const [binsRes, layoutRes, kitRes, alertRes, statsRes] = await Promise.all([
      fetch("/api/bins"),
      fetch("/api/layout"),
      fetch("/api/kit"),
      fetch("/api/alert"),
      fetch("/api/stats"),
    ]);
    if (!binsRes.ok) throw new Error("Backend error");

    const bins   = await binsRes.json();
    const layout = await layoutRes.json();
    const kit    = await kitRes.json();
    const alert  = await alertRes.json();
    const stats  = await statsRes.json();

    renderBins(bins, layout);
    renderKit(kit, bins);
    renderStatus(bins, alert);
    renderBatch(kit);
    setupMock(stats);
  } catch (err) {
    console.error("Poll error:", err);
    const msg = document.getElementById("status-msg");
    if (msg) msg.textContent = "Connection lost";
  }
}

// ── Bins ────────────────────────────────────────────
function renderBins(bins, layout) {
  const container = document.getElementById("bins");
  container.innerHTML = "";

  const byId = {};
  for (const b of bins) byId[b.id] = b;

  const layers = (layout && layout.layers) || [];
  if (layers.length === 0) {
    container.innerHTML = '<div class="bin-missing">No bins detected yet</div>';
    return;
  }

  for (const layer of layers) {
    const row = document.createElement("div");
    row.className = "bin-layer";

    const label = document.createElement("div");
    label.className = "layer-label";
    label.textContent = "L" + layer.layer;
    row.appendChild(label);

    const grid = document.createElement("div");
    grid.className = "layer-grid";
    grid.style.gridTemplateColumns = "repeat(" + Math.max(layer.row_slots, 1) + ", 1fr)";

    for (const slot of layer.bins) {
      const tile = makeBinTile(slot.id, byId[slot.id]);
      tile.style.gridColumn = (slot.slot_start + 1) + " / span " + slot.span;
      grid.appendChild(tile);
    }
    row.appendChild(grid);
    container.appendChild(row);
  }
}

function makeBinTile(binId, b) {
  const box = document.createElement("div");

  if (!b) {
    box.className = "bin missing";
    box.innerHTML = '<div class="bin-missing">MISSING BIN</div>';
    return box;
  }

  const status = binStatus(b);
  const wrong = isWrongBin(b, status);
  box.className = "bin " + status + (wrong ? " wrong" : "");

  const label = b.label || b.id;
  let inner = "";

  if (status === "missing") {
    inner += '<div class="bin-missing">MISSING BIN</div>';
  } else if (status === "not_in_bom") {
    inner += '<div class="bin-id">' + label + '</div>';
  } else {
    inner += '<div class="bin-id">' + label + '</div>';
    inner += quantityMarkup(b, status);
  }

  if (b.hand) {
    const hd = b.handedness ? b.handedness[0].toUpperCase() : "";
    inner += '<div class="hand-flag">✋ ' + hd + '</div>';
  }
  if (wrong) inner += '<div class="cross"></div>';

  if (status !== "missing" && status !== "not_in_bom") {
    inner += makeOverrideControls(b.id);
  }

  box.innerHTML = inner;
  wireOverrideControls(box, b.id);
  return box;
}

function quantityMarkup(b, status) {
  const cur = status === "overpick"
    ? '<span class="over">' + b.current + '</span>'
    : String(b.current);
  return '<div class="quantity">' + cur + '/' + b.target + '</div>';
}

// ── Manual override controls (load-cell stand-in) ───
function makeOverrideControls(binId) {
  return '<div class="bin-override" data-bin-id="' + binId + '">' +
    '<button class="ov-btn ov-minus" title="Decrement">−</button>' +
    '<button class="ov-btn ov-plus" title="Increment">+</button>' +
    '</div>';
}

function wireOverrideControls(box, binId) {
  const minus = box.querySelector(".ov-minus");
  const plus  = box.querySelector(".ov-plus");
  if (minus) minus.addEventListener("click", e => { e.stopPropagation(); overridePick(binId, -1); });
  if (plus)  plus.addEventListener("click", e => { e.stopPropagation(); overridePick(binId, +1); });
}

async function overridePick(binId, delta) {
  try {
    const res = await fetch("/api/bins/" + encodeURIComponent(binId) + "/pick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delta }),
    });
    if (res.ok) poll();
  } catch (err) { console.error("Override error:", err); }
}

// ── Kit box ─────────────────────────────────────────
function renderKit(kit, bins) {
  const placed = kit.placed || 0;
  const total  = kit.total || 0;
  const pct = total > 0 ? Math.min(100, (placed / total) * 100) : 0;

  document.getElementById("kit-placed").textContent = placed;
  document.getElementById("kit-total").textContent = total;
  document.getElementById("kit-fill").style.height = pct + "%";

  const inBom = bins.filter(b => b.in_bom && b.detected && b.target > 0);
  const allComplete = inBom.length > 0 && inBom.every(b => b.current >= b.target);
  const noFaults = !bins.some(b => {
    const s = binStatus(b);
    return s === "overpick" || isWrongBin(b, s);
  });
  const done = placed >= total && total > 0 && allComplete;

  document.getElementById("kitbox").classList.toggle("complete", done);
  document.getElementById("complete-btn").disabled = !(allComplete && noFaults);
}

// ── Batch progress ──────────────────────────────────
function renderBatch(kit) {
  const batch = kit.batch || { done: 0, target: 0 };
  document.getElementById("batch-done").textContent = batch.done;
  document.getElementById("batch-target").textContent = batch.target;
  const pct = batch.target > 0 ? (batch.done / batch.target) * 100 : 0;
  document.getElementById("batch-fill").style.width = pct + "%";
}

// ── Status bar + corrective overlay ─────────────────
function renderStatus(bins, alert) {
  const bar = document.getElementById("statusbar");
  const msgEl = document.getElementById("status-msg");
  const overlay = document.getElementById("alert-overlay");
  const overlayMsg = document.getElementById("alert-msg");

  const wrong = [];
  const over = [];
  let remaining = null;

  for (const b of bins) {
    const s = binStatus(b);
    const label = b.label || b.id;
    if (isWrongBin(b, s)) wrong.push(label);
    if (s === "overpick") over.push({ label, x: b.current - b.target });
    if (s === "incomplete") {
      const left = b.target - b.current;
      if (remaining === null || left > remaining.left) remaining = { label, left };
    }
  }

  // The full-screen overlay is driven ONLY by a backend alert (the four events).
  let message, mode;
  if (alert && alert.active) {
    message = alert.message; mode = "action";
  } else if (wrong.length) {
    message = "REMOVE HAND FROM BIN " + wrong[0]; mode = "action";
  } else if (over.length) {
    const o = over[0];
    message = "RETURN " + o.x + " ITEM" + (o.x === 1 ? "" : "S") + " TO BIN " + o.label; mode = "action";
  } else if (remaining) {
    message = "PICK " + remaining.left + " FROM BIN " + remaining.label; mode = "";
  } else {
    message = "KIT COMPLETE — ready to close"; mode = "ready";
  }

  msgEl.textContent = message;
  bar.className = "statusbar" + (mode ? " " + mode : "");

  if (alert && alert.active) {
    overlayMsg.textContent = alert.message;
    document.getElementById("alert-hint").classList.toggle("hidden", !_mockMode);
    overlay.classList.remove("hidden");
  } else {
    overlay.classList.add("hidden");
  }
}

// ── Confirm-kit flow (wired once) ───────────────────
function initCompleteFlow() {
  const btn = document.getElementById("complete-btn");
  const modal = document.getElementById("confirm-modal");
  btn.addEventListener("click", () => { if (!btn.disabled) modal.classList.remove("hidden"); });
  document.getElementById("confirm-cancel").addEventListener("click",
    () => modal.classList.add("hidden"));
  document.getElementById("confirm-proceed").addEventListener("click", async () => {
    try { await fetch("/api/kit/complete", { method: "POST" }); }
    catch (err) { console.error("Complete kit error:", err); }
    modal.classList.add("hidden");
    poll();
  });
}

// Esc: close the confirm modal if open; otherwise clear faults (mock only).
document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  const modal = document.getElementById("confirm-modal");
  if (!modal.classList.contains("hidden")) { modal.classList.add("hidden"); return; }
  if (_mockMode) fetch("/api/mock/reset-faults", { method: "POST" }).then(poll);
});

// ── Mock controls (only shown when backend reports mock mode) ──
let _mockWired = false;
let _mockMode = false;

function setupMock(stats) {
  _mockMode = !!(stats && stats.mock);
  const panel = document.getElementById("mock-controls");
  if (!_mockMode) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  if (_mockWired) return;
  _mockWired = true;

  panel.querySelectorAll(".mock-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (btn.dataset.clear) {
        await fetch("/api/mock/reset-faults", { method: "POST" });
      } else if (btn.dataset.scenario) {
        await fetch("/api/mock/alert", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scenario: btn.dataset.scenario }),
        });
      } else if (btn.dataset.hand) {
        await fetch("/api/mock/hand", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ bin: btn.dataset.hand, on: true, handedness: "right" }),
        });
      }
      poll();
    });
  });
}

// ── Start ───────────────────────────────────────────
initCompleteFlow();
poll();
setInterval(poll, POLL_INTERVAL);
