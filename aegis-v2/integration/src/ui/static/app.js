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

// ── Per-bin derivation (client-side) ──
// current = placed (verified in box); removed = taken out of the bin.
function removedOf(b) { return (b.removed != null) ? b.removed : b.current; }
// overpick = extras taken OUT of the bin (return to bin); not a red event.
function returnCount(b) { return Math.max(0, removedOf(b) - b.total); }

// Sequential FSM display state from kit.active / kit.done.
function binUiState(b, detected, kit) {
  if (!detected) return "missing";
  if (!b.using || b.total <= 0) return "not_in_bom";
  const done = (kit && kit.done) || [];
  const active = kit && kit.active;
  if (done.indexOf(b.id) !== -1) return "done";     // green, frozen
  if (active === b.id) return "active";             // the one bin being picked
  if (active) return "locked";                      // another bin active → soft-locked
  return "available";                               // idle: operator may start it
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

    renderBins(bins, layout, detectedById, kit);
    renderStatus(bins, kit);
    renderAlert(kit);
    renderEmptyBoxModal(kit);

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
function renderBins(bins, layout, detectedById, kit) {
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
      const tile = makeBinTile(slot.id, byId[slot.id], slot.detected !== false, kit);
      tile.style.gridColumn = (slot.slot_start + 1) + " / span " + slot.span;
      grid.appendChild(tile);
    }
    row.appendChild(grid);
    container.appendChild(row);
  }
}

function makeBinTile(binId, b, detected, kit) {
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

  const ui = binUiState(b, detected, kit);   // available | active | locked | done | not_in_bom
  // Hand-in-bin glow (in the bin's own colour) — clearer for the demo.
  box.className = "bin " + ui + (b.is_active ? " hand-in" : "");

  // Wrong-bin cross: the hand is in a bin the operator must NOT pick from right now —
  // not part of the job (not_in_bom), soft-locked while another bin is active, or already
  // done. The red ✗ overlay makes "don't collect here" unmistakable (CSS: .cross / .bin.wrong).
  const wrongToPick = ui === "not_in_bom" || ui === "locked" || ui === "done";
  const cross = (b.is_active && wrongToPick) ? '<div class="cross"></div>' : "";
  if (cross) box.classList.add("wrong");

  const label = b.label || b.id;
  let inner = '<div class="bin-id">' + label + '</div>';

  if (ui === "not_in_bom") { box.innerHTML = inner + cross; return box; }

  if (ui === "locked") {
    inner += '<div class="quantity muted">' + b.current + '/' + b.total + '</div>';
    box.innerHTML = inner + cross;
    return box;
  }

  // available / active / done
  if (ui === "active") {
    // Overpick: extras taken OUT of the active bin → "↩ RETURN N" above the counter.
    const ret = returnCount(b);
    if (ret > 0) inner += '<div class="return-badge">↩ RETURN ' + ret + '</div>';
  }
  inner += '<div class="quantity">' + b.current + '/' + b.total + '</div>';

  // Live load-cell weight (grams) for sensor verification.
  if (b.weight !== undefined && (b.using || Math.abs(b.weight) > 0.05)) {
    inner += '<div class="bin-weight">' + (+b.weight).toFixed(1) + ' g</div>';
  }

  if (b.is_active && b.handedness) {
    const side = b.handedness[0].toLowerCase() === "l" ? "hand-left" : "hand-right";
    inner += '<div class="hand-flag ' + side + '">✋ ' + b.handedness[0].toUpperCase() + '</div>';
  }

  // Manual override only on the active bin (the only one being picked).
  if (ui === "active") {
    inner += makeOverrideControls(b.id);
    box.innerHTML = inner + cross;
    wireOverrideControls(box, b.id);
  } else {
    box.innerHTML = inner + cross;   // `done` is wrong-to-pick → shows the cross; `available` doesn't
  }
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
function renderStatus(bins, kit) {
  const bar = document.getElementById("statusbar");
  const msgEl = document.getElementById("status-msg");
  const completeBtn = document.getElementById("complete-btn");

  const inJob = bins.filter(b => b.using && b.total > 0);
  const active = kit && kit.active;
  const complete = kit && kit.state ? !!kit.complete
    : (inJob.length > 0 && inJob.every(b => b.current === b.total && removedOf(b) === b.total));

  let message, mode;
  if (!inJob.length) {
    message = "Waiting for calibration…"; mode = "";
  } else if (kit && kit.alert) {
    // empty-kit-box also opens the popup (renderEmptyBoxModal); others also show full-screen (renderAlert).
    message = kit.alert.message; mode = "action";
  } else if (complete) {
    message = "ALL BINS COMPLETE — READY TO CLOSE"; mode = "ready";
  } else if (active) {
    const b = bins.find(x => x.id === active);
    const ret = b ? returnCount(b) : 0;
    if (ret > 0) { message = "RETURN " + ret + " TO BIN " + active; mode = "action"; }
    else { message = "PICK " + (b ? b.total - b.current : 0) + " FROM BIN " + active; mode = ""; }
  } else {
    message = "PICK FROM ANY BIN"; mode = "";
  }

  msgEl.textContent = message;
  bar.className = "statusbar" + (mode ? " " + mode : "");
  completeBtn.classList.toggle("hidden", !complete);
}

// ── Full-screen red fault overlay ───────────────────
// Shown for hard-fault events (currently load-cell-detectable: overpack-kit).
// pick/return-wrong-bin + remove-from-kit hook in here too once the backend
// emits them as kit.alert. "empty-kit-box" is routed to a popup instead (see
// renderEmptyBoxModal) — it's a normal between-sets step, not a fault.
function renderAlert(kit) {
  const overlay = document.getElementById("alert-overlay");
  if (!overlay) return;
  const alert = kit && kit.alert;
  if (alert && alert.message && alert.type !== "empty-kit-box") {
    document.getElementById("alert-msg").textContent = alert.message;
    overlay.classList.remove("hidden");
  } else {
    overlay.classList.add("hidden");
  }
}

// ── Empty-kitting-box popup ─────────────────────────
// Between sets, the pipeline blocks counts until the operator confirms (by
// button, not an automatic weight check) that the box has been emptied.
function renderEmptyBoxModal(kit) {
  const modal = document.getElementById("empty-box-modal");
  if (!modal) return;
  const alert = kit && kit.alert;
  modal.classList.toggle("hidden", !(alert && alert.type === "empty-kit-box"));
}

// ── Confirm-kit flow (wired once) ───────────────────
function initCompleteFlow() {
  const btn = document.getElementById("complete-btn");
  const modal = document.getElementById("confirm-modal");
  btn.addEventListener("click", () => { if (!btn.classList.contains("hidden")) modal.classList.remove("hidden"); });
  document.getElementById("confirm-cancel").addEventListener("click",
    () => modal.classList.add("hidden"));
  document.getElementById("confirm-proceed").addEventListener("click", async () => {
    // Close the set: the backend blocks counts (empty-kitting-box popup) until
    // the operator confirms it's been emptied, then re-tares for the next set.
    try {
      await fetch("/api/kit/complete", { method: "POST" });
    } catch (err) { console.error("Complete error:", err); }
    modal.classList.add("hidden");
    poll();
  });
}

// ── Confirm-empty flow (wired once) ─────────────────
function initEmptyBoxFlow() {
  const btn = document.getElementById("empty-box-confirm");
  btn.addEventListener("click", async () => {
    // Manual confirmation only — no automatic weight check. The pipeline tares
    // all receptors and unblocks the next set.
    try {
      await fetch("/api/kit/confirm-empty", { method: "POST" });
    } catch (err) { console.error("Confirm-empty error:", err); }
    poll();
  });
}

// ── Start ───────────────────────────────────────────
initCompleteFlow();
initEmptyBoxFlow();
poll();
setInterval(poll, POLL_INTERVAL);
