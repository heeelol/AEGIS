/**
 * AEGIS v2 — Dashboard Frontend
 * ===============================
 * Polls the FastAPI backend every 100ms and updates:
 *   - Bin grid with color-coded status
 *   - Hand tracking cards (which bin each hand is hovering over)
 *   - System stats
 *   - Error overlay
 */

const POLL_INTERVAL = 100; // ms

// ── Main poll loop ──────────────────────────────────
async function poll() {
  try {
    const [binsRes, layoutRes, handsRes, statsRes] = await Promise.all([
      fetch("/api/bins"),
      fetch("/api/layout"),
      fetch("/api/hands"),
      fetch("/api/stats"),
    ]);

    if (!binsRes.ok) throw new Error("Backend error");

    const bins   = await binsRes.json();
    const layout = await layoutRes.json();
    const hands  = await handsRes.json();
    const stats  = await statsRes.json();

    renderBins(bins, layout);
    renderHands(hands);
    renderStats(stats);

    document.getElementById("last-updated").textContent =
      "Updated: " + new Date().toLocaleTimeString();
    document.getElementById("fps-display").textContent =
      (stats.fps || 0).toFixed(0) + " FPS";

  } catch (err) {
    console.error("Poll error:", err);
    document.getElementById("last-updated").textContent = "Connection lost";
  }
}

// ── Bin grid (grouped by layer) ─────────────────────
function renderBins(bins, layout) {
  const container = document.getElementById("bins");
  container.innerHTML = "";

  // Index live bin data by id for quick lookup per slot.
  const byId = {};
  for (const bin of bins) byId[bin.id] = bin;

  const layers = (layout && layout.layers) || [];

  if (layers.length === 0) {
    container.innerHTML = '<div class="no-hands">No bins detected yet</div>';
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
    // Lay the row out as a slot track; each bin spans its declared slots.
    grid.style.gridTemplateColumns = "repeat(" + Math.max(layer.row_slots, 1) + ", 1fr)";

    for (const slot of layer.bins) {
      const boxEl = makeBinBox(slot.id, byId[slot.id]);
      boxEl.style.gridColumn = (slot.slot_start + 1) + " / span " + slot.span;
      grid.appendChild(boxEl);
    }

    row.appendChild(grid);
    container.appendChild(row);
  }
}

function makeBinBox(binId, bin) {
  const box = document.createElement("div");

  // Slot in the layout but no live data yet → render as a grey placeholder.
  if (!bin) {
    box.className = "bin grey";
    box.innerHTML = '<div class="bin-id">' + binId + '</div>'
                  + makeOverrideControls(binId);
    wireOverrideControls(box, binId);
    return box;
  }

  const status = String(bin.status).trim().toLowerCase();
  box.className = "bin " + status;

  let inner = '<div class="bin-id">' + bin.id + '</div>';

  if (status !== "grey") {
    if (bin.total > 0) {
      inner += '<div class="quantity">' + bin.current + '/' + bin.total + '</div>';
    } else {
      inner += '<div class="quantity">' + bin.current + '</div>';
    }
    if (bin.weight) {
      inner += '<div class="bin-weight">' + Math.round(bin.weight) + 'g</div>';
    }
  } else {
    inner += '<div class="quantity">' + bin.current + '</div>';
  }

  if (bin.is_active && bin.handedness) {
    inner += '<div class="bin-badge" style="background:#3b82f6;color:#fff;">'
           + bin.handedness.toUpperCase() + '</div>';
  }

  inner += makeOverrideControls(bin.id);

  box.innerHTML = inner;
  wireOverrideControls(box, bin.id);
  return box;
}

// ── Manual load-cell override controls ──────────────
function makeOverrideControls(binId) {
  return (
    '<div class="bin-override" data-bin-id="' + binId + '">' +
      '<button class="ov-btn ov-minus" title="Decrement pick count">−</button>' +
      '<button class="ov-btn ov-plus" title="Increment pick count">+</button>' +
    '</div>'
  );
}

function wireOverrideControls(box, binId) {
  const minus = box.querySelector(".ov-minus");
  const plus  = box.querySelector(".ov-plus");
  if (minus) minus.addEventListener("click", function(e) {
    e.stopPropagation();
    overridePick(binId, -1);
  });
  if (plus) plus.addEventListener("click", function(e) {
    e.stopPropagation();
    overridePick(binId, +1);
  });
}

async function overridePick(binId, delta) {
  try {
    const res = await fetch("/api/bins/" + encodeURIComponent(binId) + "/pick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delta: delta }),
    });
    if (!res.ok) {
      console.warn("Override failed:", binId, delta, res.status);
      return;
    }
    // Refresh immediately so the operator sees the change without waiting for the next poll.
    poll();
  } catch (err) {
    console.error("Override error:", err);
  }
}

// ── Hand cards ──────────────────────────────────────
function renderHands(hands) {
  const container = document.getElementById("hands");
  container.innerHTML = "";

  if (hands.length === 0) {
    container.innerHTML = '<div class="no-hands">No hands detected</div>';
    return;
  }

  for (const hand of hands) {
    const card = document.createElement("div");
    card.className = "hand-card";
    if (hand.assigned_bin) card.classList.add("hovering");

    card.innerHTML =
      '<div class="hand-label">' + hand.handedness + ' hand</div>' +
      '<div class="hand-detail">Position: (' +
        Math.round(hand.x) + ', ' + Math.round(hand.y) + ')</div>' +
      '<div class="hand-detail">Hovering: ' +
        (hand.assigned_bin || '—') + '</div>';

    container.appendChild(card);
  }
}

// ── Stats ───────────────────────────────────────────
function renderStats(stats) {
  document.getElementById("stat-frames").textContent =
    (stats.frame_count || 0).toLocaleString();
  document.getElementById("stat-uptime").textContent =
    formatUptime(stats.uptime_seconds || 0);
  document.getElementById("stat-bins").textContent = stats.num_bins || 0;
  document.getElementById("stat-hands").textContent = stats.num_hands || 0;
}

function formatUptime(seconds) {
  if (seconds < 60) return Math.floor(seconds) + "s";
  if (seconds < 3600) return Math.floor(seconds / 60) + "m " + (Math.floor(seconds) % 60) + "s";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h + "h " + m + "m";
}

// A hand in a not-in-use bin (status "wrong_bin") lights up that bin directly
// via the .bin.wrong_bin CSS glow — no full-screen overlay. The illumination
// clears the instant the hand leaves, since status is recomputed every poll.

// ── Start polling ───────────────────────────────────
// Self-scheduling loop: wait for each poll to finish before scheduling the
// next. At a 100ms interval a plain setInterval could stack overlapping
// requests if one round-trip ran slow; this keeps at most one poll in flight.
(async function pollLoop() {
  await poll();
  setTimeout(pollLoop, POLL_INTERVAL);
})();
