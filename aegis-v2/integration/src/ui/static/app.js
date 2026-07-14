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
    const [binsRes, layoutRes, statsRes, kitRes, cycleRes] = await Promise.all([
      fetch("/api/bins"),
      fetch("/api/layout"),
      fetch("/api/stats"),
      fetch("/api/kit"),
      fetch("/api/cycle"),
    ]);
    if (!binsRes.ok) throw new Error("Backend error");

    const bins = await binsRes.json();
    const layout = await layoutRes.json();
    const stats  = await statsRes.json();
    const kit    = kitRes.ok ? await kitRes.json() : {};
    const cycle  = cycleRes.ok ? await cycleRes.json() : {};

    const detectedById = {};
    for (const layer of (layout.layers || [])) {
      for (const slot of layer.bins) detectedById[slot.id] = slot.detected !== false;
    }

    renderBins(bins, layout, detectedById, kit);
    renderAlert(kit);
    renderEmptyBoxModal(kit);
    renderCycle(cycle);

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
    label.textContent = "LAYER " + (layer.layer + 1);   // 1-indexed for operators
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
  // Fault highlight: the specific bin named by an active fault glows directly,
  // so the operator doesn't have to match a bin ID in a sentence to a physical
  // tile (paired with the banner's #alert-bin badge — see style.css .bin.fault).
  const isFaultBin = kit && kit.alert && kit.alert.bin === binId;
  box.className = "bin " + ui + (b.is_active ? " hand-in" : "") + (isFaultBin ? " fault" : "");

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

  // (Left/right hand chip removed — the bold hand-in halo on the whole tile is
  // the single, unmistakable "your hand is here" cue.)

  // (Manual +/- override controls removed — counts come from the load cells.)
  box.innerHTML = inner + cross;   // `done` is wrong-to-pick → shows the cross; `available` doesn't
  return box;
}

// ── Full-screen red fault banner ────────────────────
// One big keyword (mapped from the fault type) with the corrective action as a
// subtitle, plus the specific bin badge. Covers all four hard faults; each
// auto-clears once the backend's kit.alert goes back to null.
const FAULT_KEYWORDS = {
  "overpack-kit":        "OVERPACK",
  "pick-from-wrong-bin": "PICKED FROM WRONG BIN",
  "return-to-wrong-bin": "RETURNED TO WRONG BIN",
  "out-of-sequence":     "OUT OF SEQUENCE",
};
// Minimum time the banner stays visible once shown. A fault corrected within a
// single poll cycle would otherwise flash for ~1 poll interval — too fast to
// notice, even though it was genuinely detected.
const ALERT_MIN_VISIBLE_MS = 900;
let alertHideAt = 0;

function renderAlert(kit) {
  const overlay = document.getElementById("alert-overlay");
  if (!overlay) return;
  const alert = kit && kit.alert;
  const binEl = document.getElementById("alert-bin");
  const now = Date.now();
  if (alert && alert.type && FAULT_KEYWORDS[alert.type]) {
    document.getElementById("alert-keyword").textContent = FAULT_KEYWORDS[alert.type];
    document.getElementById("alert-sub").textContent = alert.message || "";
    // Large, separate at-a-glance target for the specific bin (friendly "BIN N").
    if (binEl) {
      const label = alert.bin_label || alert.bin;
      if (label) {
        binEl.textContent = label;
        binEl.classList.remove("hidden");
      } else {
        binEl.classList.add("hidden");
      }
    }
    overlay.classList.remove("hidden");
    alertHideAt = now + ALERT_MIN_VISIBLE_MS;
  } else if (now >= alertHideAt) {
    overlay.classList.add("hidden");
    if (binEl) binEl.classList.add("hidden");
  }
}

// ── Empty-kitting-box popup ─────────────────────────
// Appears automatically once every bin is green (kit.complete) — no Complete
// button. A fault takes priority: if the operator makes a mistake before
// emptying (e.g. one more item), the fault banner shows and this popup hides;
// correcting it brings the popup back. Confirming empties + advances the set.
function renderEmptyBoxModal(kit) {
  const modal = document.getElementById("empty-box-modal");
  if (!modal) return;
  const show = kit && kit.complete && !kit.alert;
  modal.classList.toggle("hidden", !show);
}

// ── Sets-remaining progress ─────────────────────────
// Surfaces the work-order cycle ({set_number, total_sets, complete} from
// /api/cycle) that the backend already tracks. Answers the operators'
// unprompted "how many sets left?" — a chip in the header plus a slim bar
// that fills with completed sets. Both stay hidden until a work order loads.
function renderCycle(cycle) {
  const chip = document.getElementById("set-progress");
  const bar  = document.getElementById("set-progress-bar");
  const fill = document.getElementById("set-progress-fill");
  const total = cycle && cycle.total_sets;
  if (!total) {
    chip.classList.add("hidden");
    bar.classList.add("hidden");
    return;
  }
  const n = cycle.set_number || 0;
  chip.textContent = "SET " + n + " / " + total;
  chip.classList.remove("hidden");
  // Fill = sets fully finished (n-1 in progress, or all when the cycle is done).
  const doneSets = cycle.complete ? total : Math.max(0, n - 1);
  fill.style.width = (100 * doneSets / total) + "%";
  bar.classList.remove("hidden");
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
initEmptyBoxFlow();
poll();
setInterval(poll, POLL_INTERVAL);
