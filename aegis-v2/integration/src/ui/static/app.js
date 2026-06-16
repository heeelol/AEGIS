/**
 * AEGIS — Kitting Operator Frontend
 * ==================================
 * The bins are the whole story. A bin shows placed/target and turns green ONLY
 * when it's truly finished — the box has its items AND no extras are still held
 * (placed == target AND removed == target). The picking/returning mechanics are
 * never shown on the tile; the one instruction lives in the status bar.
 *
 * Data contract:
 *   /api/bins  -> [{id,label,in_bom,detected,target,removed,placed,hand,handedness}]
 *   /api/kit   -> {batch:{done,target}}   (box sensor stays in backend)
 *   /api/alert -> {active,message}
 *   /api/layout-> {layers:[{layer,row_slots,bins:[{id,slot_start,span}]}]}
 */

const POLL_INTERVAL = 1000; // ms

// ── Per-bin derivation ──────────────────────────────
function binStatus(b) {
  if (!b.detected) return "missing";
  if (!b.in_bom || b.target <= 0) return "not_in_bom";
  // Done only when the box has its items AND nothing extra is still in hand.
  if (b.placed >= b.target && b.removed <= b.target) return "complete";
  return "incomplete";
}
function toPlace(b)  { return Math.max(0, b.target - b.placed); }
function needsReturn(b) { return b.placed >= b.target && b.removed > b.target; }
function returnAmt(b) { return Math.max(0, b.removed - b.target); }
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
    // placed in box / target — the only number. No picked/return/in-hand clutter.
    inner += '<div class="bin-id">' + label + '</div>';
    inner += '<div class="quantity">' + Math.min(b.placed, b.target) + '/' + b.target + '</div>';
  }

  if (b.hand) {
    const hd = b.handedness ? b.handedness[0].toUpperCase() : "";
    inner += '<div class="hand-flag">✋ ' + hd + '</div>';
  }
  if (wrong) inner += '<div class="cross"></div>';

  if (status !== "missing" && status !== "not_in_bom") {
    inner += makeBinControls(b.id);
  }

  box.innerHTML = inner;
  wireBinControls(box, b.id);
  return box;
}

// ── Per-bin controls: grab / place / return ─────────
function makeBinControls(binId) {
  return '<div class="bin-ctl" data-bin-id="' + binId + '">' +
    '<button class="ctl-btn ctl-grab"   title="Take one from bin">grab</button>' +
    '<button class="ctl-btn ctl-place"  title="Place one in kit box">place</button>' +
    '<button class="ctl-btn ctl-return" title="Return one to bin">return</button>' +
    '</div>';
}

function wireBinControls(box, binId) {
  const map = { ".ctl-grab": "grab", ".ctl-place": "place", ".ctl-return": "return" };
  for (const [sel, action] of Object.entries(map)) {
    const el = box.querySelector(sel);
    if (el) el.addEventListener("click", e => { e.stopPropagation(); binAction(binId, action); });
  }
}

async function binAction(binId, action) {
  try {
    const res = await fetch("/api/bins/" + encodeURIComponent(binId) + "/" + action, { method: "POST" });
    if (res.ok) poll();
  } catch (err) { console.error("Bin action error:", err); }
}

// ── Batch progress ──────────────────────────────────
function renderBatch(kit) {
  const batch = (kit && kit.batch) || { done: 0, target: 0 };
  document.getElementById("batch-done").textContent = batch.done;
  document.getElementById("batch-target").textContent = batch.target;
  const pct = batch.target > 0 ? (batch.done / batch.target) * 100 : 0;
  document.getElementById("batch-fill").style.width = pct + "%";
}

// ── Status bar + Complete button + overlay ──────────
function renderStatus(bins, alert) {
  const bar = document.getElementById("statusbar");
  const msgEl = document.getElementById("status-msg");
  const overlay = document.getElementById("alert-overlay");
  const overlayMsg = document.getElementById("alert-msg");
  const completeBtn = document.getElementById("complete-btn");

  const wrong = [];
  const places = [];   // bins still needing items in the box
  const returns = [];  // bins that finished placing but hold extras
  const inBom = [];

  for (const b of bins) {
    const s = binStatus(b);
    const label = b.label || b.id;
    if (s === "missing" || s === "not_in_bom") continue;
    inBom.push(b);
    if (isWrongBin(b, s)) wrong.push(label);
    if (s === "incomplete" && toPlace(b) > 0) places.push({ label, n: toPlace(b) });
    if (needsReturn(b)) returns.push({ label, n: returnAmt(b) });
  }

  const allDone = inBom.length > 0 && inBom.every(b => binStatus(b) === "complete");

  // One instruction. Placing first (productive); returns surface once placing is done.
  let message, mode;
  if (alert && alert.active) {
    message = alert.message; mode = "action";
  } else if (wrong.length) {
    message = "REMOVE HAND FROM BIN " + wrong[0]; mode = "action";
  } else if (places.length) {
    const p = places.sort((a, b) => b.n - a.n)[0];
    message = "PLACE " + p.n + " FROM BIN " + p.label; mode = "";
  } else if (returns.length) {
    const r = returns[0];
    message = "RETURN " + r.n + " ITEM" + (r.n === 1 ? "" : "S") + " TO BIN " + r.label; mode = "action";
  } else if (allDone) {
    message = "ALL BINS COMPLETE — READY TO CLOSE"; mode = "ready";
  } else {
    message = "Checking bins…"; mode = "";
  }

  msgEl.textContent = message;
  bar.className = "statusbar" + (mode ? " " + mode : "");

  // Complete button appears only when every bin is green and there are no faults.
  const ready = allDone && !wrong.length && !returns.length && !(alert && alert.active);
  completeBtn.classList.toggle("hidden", !ready);

  // Full-screen overlay only for the backend alert events.
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
  btn.addEventListener("click", () => { if (!btn.classList.contains("hidden")) modal.classList.remove("hidden"); });
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
