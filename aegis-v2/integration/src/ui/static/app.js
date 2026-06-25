/**
 * AEGIS — Kitting Operator Frontend
 * ==================================
 * Runs on the live dashboard.py backend. Bins fill toward green; the one
 * instruction lives in the status bar; the Complete button appears when every
 * in-job bin is at its target. Pick-based: a bin is green when current == total
 * (load-cell / FSM driven). Layout is slot-proportional (6 top + 3 bottom).
 *
 * Consumes:
 *   /api/bins   -> [{id,label,layer,col,current,total,using,is_active,handedness,...}]
 *   /api/layout -> {layers:[{layer,row_slots,bins:[{id,slot_start,span,detected}]}]}
 *   /api/stats  -> {fps,...}
 *   /api/bins/{id}/pick  (load-cell stand-in: {delta} / {count})
 */

const POLL_INTERVAL = 250; // ms

// ── Per-bin derivation (client-side, from raw fields + detected) ──
function binStatus(b, detected) {
  if (!detected) return "missing";
  if (!b.using || b.total <= 0) return "not_in_bom";
  if (b.current > b.total) return "overpick";
  if (b.current >= b.total) return "complete";
  return "incomplete";
}
function isWrongBin(b, status) {
  return b.is_active && (status === "not_in_bom" || status === "complete");
}

// ── Main poll loop ──────────────────────────────────
async function poll() {
  try {
    const [binsRes, layoutRes, statsRes, kitRes] = await Promise.all([
      fetch("/api/bins"),
      fetch("/api/layout"),
      fetch("/api/stats"),
      fetch("/api/kit"),
    ]);
    if (!binsRes.ok) throw new Error("Backend error");

    const bins   = await binsRes.json();
    const layout = await layoutRes.json();
    const stats  = await statsRes.json();
    const kit    = kitRes.ok ? await kitRes.json() : {};

    const detectedById = {};
    for (const layer of (layout.layers || [])) {
      for (const slot of layer.bins) detectedById[slot.id] = slot.detected !== false;
    }

    renderBins(bins, layout, detectedById);
    renderStatus(bins, detectedById, kit);

    document.getElementById("last-updated").textContent =
      "Updated " + new Date().toLocaleTimeString();
    document.getElementById("fps-display").textContent =
      (stats.fps || 0).toFixed(0) + " FPS";
  } catch (err) {
    console.error("Poll error:", err);
    document.getElementById("last-updated").textContent = "Connection lost";
  }
}

// ── Bins (slot-proportional grid) ───────────────────
function renderBins(bins, layout, detectedById) {
  const container = document.getElementById("bins");
  container.innerHTML = "";

  const byId = {};
  for (const b of bins) byId[b.id] = b;

  const layers = (layout && layout.layers) || [];
  if (layers.length === 0) {
    container.innerHTML = '<div class="bin-missing">No bins yet — press 1 to calibrate, 2 to load the kit</div>';
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
      const tile = makeBinTile(slot.id, byId[slot.id], slot.detected !== false);
      tile.style.gridColumn = (slot.slot_start + 1) + " / span " + slot.span;
      grid.appendChild(tile);
    }
    row.appendChild(grid);
    container.appendChild(row);
  }
}

function makeBinTile(binId, b, detected) {
  const box = document.createElement("div");

  if (!detected) {
    box.className = "bin missing";
    box.innerHTML = '<div class="bin-missing">MISSING BIN</div>';
    return box;
  }
  if (!b) {
    box.className = "bin not_in_bom";
    box.innerHTML = '<div class="bin-id">' + binId + '</div>';
    return box;
  }

  const status = binStatus(b, detected);
  const wrong = isWrongBin(b, status);
  box.className = "bin " + status + (wrong ? " wrong" : "");

  const label = b.label || b.id;
  let inner = '<div class="bin-id">' + label + '</div>';

  if (status !== "not_in_bom") {
    // On overpick, show a small "return N" indicator above the counter.
    if (status === "overpick") {
      inner += '<div class="return-badge">↩ RETURN ' + (b.current - b.total) + '</div>';
    }
    const cur = status === "overpick"
      ? '<span class="over">' + b.current + '</span>'
      : String(b.current);
    inner += '<div class="quantity">' + cur + '/' + b.total + '</div>';
  }

  // Live load-cell weight (grams) — shown for bins with a cell so the sensor can
  // be verified in real time. Bins without a cell read 0 and aren't in the job,
  // so they stay clean.
  if (b.weight !== undefined && (b.using || Math.abs(b.weight) > 0.05)) {
    inner += '<div class="bin-weight">' + (+b.weight).toFixed(1) + ' g</div>';
  }

  if (b.is_active && b.handedness) {
    // Place the flag on the side the hand should approach from: left hand → left.
    const side = b.handedness[0].toLowerCase() === "l" ? "hand-left" : "hand-right";
    inner += '<div class="hand-flag ' + side + '">✋ ' + b.handedness[0].toUpperCase() + '</div>';
  }
  if (wrong) inner += '<div class="cross"></div>';

  inner += makeOverrideControls(b.id);
  box.innerHTML = inner;
  wireOverrideControls(box, b.id);
  return box;
}

// ── Manual override (load-cell stand-in) ────────────
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

// ── Status bar + Complete button ────────────────────
function renderStatus(bins, detectedById, kit) {
  const bar = document.getElementById("statusbar");
  const msgEl = document.getElementById("status-msg");
  const completeBtn = document.getElementById("complete-btn");

  const wrong = [];
  const over = [];
  const inJob = [];
  let pickNext = null;

  for (const b of bins) {
    const detected = detectedById[b.id] !== false;
    const s = binStatus(b, detected);
    if (s === "missing" || s === "not_in_bom") continue;
    inJob.push(b);
    const label = b.label || b.id;
    if (isWrongBin(b, s)) wrong.push(label);
    if (s === "overpick") over.push({ label, n: b.current - b.total });
    if (s === "incomplete") {
      const left = b.total - b.current;
      if (pickNext === null || left > pickNext.left) pickNext = { label, left };
    }
  }

  const allDone = inJob.length > 0 && inJob.every(b => b.current === b.total);

  let message, mode;
  if (!inJob.length) {
    message = "Waiting for calibration…"; mode = "";
  } else if (wrong.length) {
    message = "REMOVE HAND FROM BIN " + wrong[0]; mode = "action";
  } else if (over.length) {
    const o = over[0];
    message = "RETURN " + o.n + " ITEM" + (o.n === 1 ? "" : "S") + " TO BIN " + o.label; mode = "action";
  } else if (pickNext) {
    message = "PICK " + pickNext.left + " FROM BIN " + pickNext.label; mode = "";
  } else if (kit && kit.state && !kit.complete) {
    // Bins met their targets but the box weight hasn't confirmed yet.
    const bg = (+kit.box_grams || 0).toFixed(0), eg = (+kit.expected_grams || 0).toFixed(0);
    message = "VERIFY KITTING BOX — " + bg + " / " + eg + " g"; mode = "";
  } else {
    message = "ALL BINS COMPLETE — READY TO CLOSE"; mode = "ready";
  }

  // Completion is authoritative from the kit FSM when load cells are live
  // (both targets met AND box total matches); otherwise fall back to bin counts.
  const kitComplete = (kit && kit.state)
    ? !!kit.complete
    : (allDone && !wrong.length && !over.length);

  msgEl.textContent = message;
  bar.className = "statusbar" + (mode ? " " + mode : "");
  completeBtn.classList.toggle("hidden", !(kitComplete && !wrong.length));
}

// ── Confirm-kit flow (wired once) ───────────────────
function initCompleteFlow() {
  const btn = document.getElementById("complete-btn");
  const modal = document.getElementById("confirm-modal");
  btn.addEventListener("click", () => { if (!btn.classList.contains("hidden")) modal.classList.remove("hidden"); });
  document.getElementById("confirm-cancel").addEventListener("click",
    () => modal.classList.add("hidden"));
  document.getElementById("confirm-proceed").addEventListener("click", async () => {
    // Close the kit: the backend re-tares all 3 receptors for the next run.
    try {
      await fetch("/api/kit/complete", { method: "POST" });
    } catch (err) { console.error("Complete error:", err); }
    modal.classList.add("hidden");
    poll();
  });
}

// ── Start ───────────────────────────────────────────
initCompleteFlow();
poll();
setInterval(poll, POLL_INTERVAL);
