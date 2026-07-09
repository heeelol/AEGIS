# Load-Cell-Driven Pick Count (bin_0_5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ESP32 load cell drive bin_0_5's dashboard pick count automatically — removing AMPs updates "current / target" with no manual input.

**Architecture:** The serial → state → dashboard path already exists. We add (1) a host-side `bin_remap` in `LoadCellReader` so the firmware's `bin_0_0` key becomes `bin_0_5` downstream, and (2) a thin conversion in `Pipeline` that turns the mapped bin's weight into a pick count via the existing `InventoryTracker` and pushes it to `PipelineState.set_pick_count`. Authoritative for mapped bins, guarded so a disconnect never clobbers the count.

**Tech Stack:** Python 3.13, pytest 9, pyyaml, pyserial (runtime only; tests use no serial). Run tests from `C:\Users\chenx\Documents\TEnterns\aegis-v2`.

## Global Constraints

- All commands run from repo root `C:\Users\chenx\Documents\TEnterns\aegis-v2`.
- Test invocation: `python -m pytest <path> -q` (matches existing `integration/tests/`).
- Tests are pure — no camera, no serial, no cv2 (mirror `test_finger_vote.py`'s style: manual `sys.path` insert + direct imports).
- bin id format is `bin_{row}_{col}`. The load cell physically maps to **bin_0_5**; firmware emits **bin_0_0**.
- AMP unit weight = **3.8 g** (one AMP = 3.8 g, confirmed live). Item key: `amp` (lowercase, matches existing `bolt_m6` style).
- Live port: **COM5** (Silicon Labs CP210x, driver installed 2026-06-17).
- Commit staging: stage ONLY this feature's files (the repo has unrelated pre-existing modifications to `FUTURE_TASKS.md` and `settings.yaml` — do not bundle them; `settings.yaml` is edited in Task 3 and staged there).

---

### Task 1: `bin_remap` in LoadCellReader

Rename incoming bin ids at cache time so every downstream consumer (weights, layout, inventory, dashboard) speaks `bin_0_5`, never the firmware's `bin_0_0`.

**Files:**
- Modify: `integration/src/sensing/loadcell.py` (add `self._remap`, `_apply_remap`, call it in `_read_loop`)
- Test: `integration/tests/test_loadcell_remap.py` (create)

**Interfaces:**
- Consumes: `LoadCellReader._parse_line(raw: bytes) -> dict[str,float] | None` (existing, static).
- Produces: `LoadCellReader._apply_remap(weights: dict[str,float]) -> dict[str,float]` — returns a new dict with each key replaced by `self._remap.get(key, key)`; identity when remap empty or weights empty. `self._remap` is read from `config["bin_remap"]` (default `{}`). `get_weights()` / `get_layout()` return remapped ids because the cache (`self._weights`) is stored already-remapped.

- [ ] **Step 1: Write the failing test**

Create `integration/tests/test_loadcell_remap.py`:

```python
"""Unit tests for LoadCellReader's bin_remap (firmware key -> canonical bin id).

Pure — no serial. enabled defaults to False so __init__ opens no port.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.sensing.loadcell import LoadCellReader  # noqa: E402


def _reader(remap=None):
    cfg = {}
    if remap is not None:
        cfg["bin_remap"] = remap
    return LoadCellReader(cfg)  # enabled False -> no serial port opened


def test_apply_remap_renames_known_key():
    r = _reader({"bin_0_0": "bin_0_5"})
    assert r._apply_remap({"bin_0_0": -7.6}) == {"bin_0_5": -7.6}


def test_apply_remap_passes_through_unknown_key():
    r = _reader({"bin_0_0": "bin_0_5"})
    assert r._apply_remap({"bin_1_2": -1.0}) == {"bin_1_2": -1.0}


def test_apply_remap_identity_when_no_remap():
    r = _reader()
    assert r._apply_remap({"bin_0_0": -7.6}) == {"bin_0_0": -7.6}


def test_get_weights_and_layout_reflect_remap():
    # Simulate the read loop: parse a real firmware line, then cache it remapped.
    r = _reader({"bin_0_0": "bin_0_5"})
    parsed = LoadCellReader._parse_line(b'{"bins":{"bin_0_0":-7.6}}\n')
    r._weights = r._apply_remap(parsed)
    assert r.get_weights() == {"bin_0_5": -7.6}
    layout = r.get_layout()
    # bin_0_5 -> row 0, col 5 -> 6 bins in row 0
    assert layout.bins_per_layer == {0: 6}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest integration/tests/test_loadcell_remap.py -q`
Expected: FAIL — `AttributeError: 'LoadCellReader' object has no attribute '_apply_remap'`.

- [ ] **Step 3: Write minimal implementation**

In `integration/src/sensing/loadcell.py`, inside `__init__` (after `self._stale_after = ...`), add:

```python
        # Optional {firmware_bin_id: canonical_bin_id} rename, applied to every
        # reading before it is cached, so all downstream consumers (weights,
        # layout, inventory, dashboard) see the canonical id.
        self._remap: dict[str, str] = dict(self._config.get("bin_remap") or {})
```

Add the helper method (place it next to `_parse_line`):

```python
    def _apply_remap(self, weights: dict[str, float]) -> dict[str, float]:
        """Rename bin ids per ``self._remap``; identity when remap/weights empty."""
        if not self._remap or not weights:
            return weights
        return {self._remap.get(bin_id, bin_id): grams
                for bin_id, grams in weights.items()}
```

In `_read_loop`, change the cache step so the remap is applied before storing:

```python
            weights = self._parse_line(raw)
            if weights is not None:
                weights = self._apply_remap(weights)
                with self._lock:
                    self._weights = weights
                    self._last_rx = time.time()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest integration/tests/test_loadcell_remap.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add integration/src/sensing/loadcell.py integration/tests/test_loadcell_remap.py
git commit -m "feat(v2): loadcell bin_remap — rename firmware bin id at cache time"
```

---

### Task 2: Weight → pick count wiring in Pipeline

Convert the mapped bin's (remapped) weight into a pick count via `InventoryTracker` and push it to `PipelineState`, guarded so a disconnected cell never overwrites.

**Files:**
- Modify: `integration/src/pipeline.py` (add `derive_pick_counts` free function; build `self._inventory` in `_init_loadcells`; add `_apply_loadcell_counts`; call it at init and in the poll block)
- Modify: `integration/config/inventory.yaml` (add `amp` item + `bin_0_5` mapping)
- Test: `integration/tests/test_loadcell_count.py` (create)

**Interfaces:**
- Consumes: `InventoryTracker.items_taken(weights: dict[str,float]) -> dict[str,int]` (existing — only returns bins present in inventory.yaml, count clamped ≥0). `LoadCellReader.get_weights()` / `.is_connected()` (existing). `PipelineState.set_pick_count(bin_id, count) -> int | None` (existing).
- Produces: `derive_pick_counts(weights, tracker, connected) -> dict[str,int]` — returns `tracker.items_taken(weights)` when `connected` is True, else `{}`. `Pipeline._apply_loadcell_counts()` — applies those counts to `self._state` via `set_pick_count`.

- [ ] **Step 1: Write the failing test**

Create `integration/tests/test_loadcell_count.py`:

```python
"""Unit tests for load-cell -> pick-count wiring.

Pure — no serial, no camera. Uses a temp inventory.yaml and a fake reader.
Pipeline._apply_loadcell_counts is exercised via object.__new__ (we set only
the attributes the method touches; __init__ would open a camera/config).
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.pipeline import Pipeline, derive_pick_counts  # noqa: E402
from integration.src.sensing.inventory import InventoryTracker  # noqa: E402
from integration.src.ui.state import PipelineState  # noqa: E402


def _tracker(tmp_path):
    inv = tmp_path / "inventory.yaml"
    inv.write_text(
        "items:\n"
        "  amp: { unit_g: 3.8 }\n"
        "bins:\n"
        "  bin_0_5: amp\n"
    )
    return InventoryTracker(str(inv))


class _FakeReader:
    def __init__(self, weights, connected):
        self._weights, self._connected = weights, connected

    def get_weights(self):
        return dict(self._weights)

    def is_connected(self):
        return self._connected


def test_derive_counts_when_connected(tmp_path):
    tracker = _tracker(tmp_path)
    # -7.6 g / 3.8 g = 2 AMPs taken
    assert derive_pick_counts({"bin_0_5": -7.6}, tracker, True) == {"bin_0_5": 2}


def test_derive_counts_empty_when_disconnected(tmp_path):
    tracker = _tracker(tmp_path)
    assert derive_pick_counts({"bin_0_5": -7.6}, tracker, False) == {}


def test_derive_counts_clamps_and_ignores_unmapped(tmp_path):
    tracker = _tracker(tmp_path)
    # small positive (noise) -> 0; unmapped bin not in inventory -> omitted
    assert derive_pick_counts({"bin_0_5": 1.0, "bin_9_9": -100.0}, tracker, True) \
        == {"bin_0_5": 0}


def _pipeline_with(state, reader, tracker):
    p = object.__new__(Pipeline)        # bypass __init__ (no camera/config)
    p._state, p._loadcells, p._inventory = state, reader, tracker
    return p


def test_apply_counts_sets_state_when_connected(tmp_path):
    state = PipelineState()
    state.update_bins({"bin_0_5": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}})
    p = _pipeline_with(state, _FakeReader({"bin_0_5": -7.6}, True), _tracker(tmp_path))

    p._apply_loadcell_counts()

    current = {b["id"]: b["current"] for b in state.get_bins()}
    assert current["bin_0_5"] == 2


def test_apply_counts_noop_when_disconnected(tmp_path):
    state = PipelineState()
    state.update_bins({"bin_0_5": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}})
    state.set_pick_count("bin_0_5", 4)  # pre-existing manual value
    p = _pipeline_with(state, _FakeReader({"bin_0_5": -7.6}, False), _tracker(tmp_path))

    p._apply_loadcell_counts()

    current = {b["id"]: b["current"] for b in state.get_bins()}
    assert current["bin_0_5"] == 4  # untouched — guard held
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest integration/tests/test_loadcell_count.py -q`
Expected: FAIL — `ImportError: cannot import name 'derive_pick_counts'`.

- [ ] **Step 3: Write minimal implementation**

In `integration/src/pipeline.py`, add a module-level function (place it just above `class Pipeline`):

```python
def derive_pick_counts(weights: dict, tracker, connected: bool) -> dict:
    """Pick counts to push to state from load-cell weights.

    Returns ``{bin_id: count}`` for inventory-mapped bins when the cell is
    connected, else ``{}`` (the connected-guard: a dropped link must not
    overwrite counts). ``tracker.items_taken`` already clamps at >= 0 and
    only includes bins present in inventory.yaml.
    """
    if not connected:
        return {}
    return tracker.items_taken(weights)
```

In `_init_loadcells`, after `self._loadcells = LoadCellReader(lc_cfg)`, build the tracker and push an initial count:

```python
        from integration.src.sensing import InventoryTracker
        self._inventory = InventoryTracker()        # loads config/inventory.yaml
```

Then, still in `_init_loadcells`, after the existing `self._state.update_loadcells(...)` line, add:

```python
        self._apply_loadcell_counts()
```

Add the method (place it right after `_init_loadcells`):

```python
    def _apply_loadcell_counts(self) -> None:
        """Drive mapped bins' pick counts from the latest load-cell weights.

        Authoritative for inventory-mapped bins while the cell is connected;
        the connected-guard in ``derive_pick_counts`` leaves counts alone when
        the link drops. Bins absent from inventory.yaml are never touched.
        """
        if self._loadcells is None or self._inventory is None:
            return
        counts = derive_pick_counts(
            self._loadcells.get_weights(),
            self._inventory,
            self._loadcells.is_connected(),
        )
        for bin_id, count in counts.items():
            self._state.set_pick_count(bin_id, count)
```

Add the `_inventory` attribute declaration in `__init__` (next to `self._loadcells = None`):

```python
        self._inventory = None
```

In `_main_loop`, extend the existing every-30-frames poll block so the counts refresh with the weights:

```python
                if self._loadcells is not None:
                    self._state.update_loadcells(
                        self._loadcells.get_layout(),
                        self._loadcells.get_weights(),
                    )
                    self._apply_loadcell_counts()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest integration/tests/test_loadcell_count.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Add the live inventory mapping**

Edit `integration/config/inventory.yaml` — add the `amp` item and `bin_0_5` mapping (leave existing `bolt_m6`/`bin_0_0` entries as-is):

```yaml
items:
  bolt_m6:   { unit_g: 13.9 }
  washer_m6: { unit_g: 1.1 }
  amp:       { unit_g: 3.8 }

bins:
  bin_0_0: bolt_m6
  bin_0_5: amp
```

- [ ] **Step 6: Run the full sensing test set to confirm nothing regressed**

Run: `python -m pytest integration/tests/test_loadcell_remap.py integration/tests/test_loadcell_count.py -q`
Expected: PASS (9 passed total).

- [ ] **Step 7: Commit**

```bash
git add integration/src/pipeline.py integration/tests/test_loadcell_count.py integration/config/inventory.yaml
git commit -m "feat(v2): drive bin_0_5 pick count from load cell (AMP @ 3.8g)"
```

---

### Task 3: Enable the live link + verify on hardware

Turn the load cell on in config and confirm the real loop end-to-end. No unit test — this task's deliverable is observed live behavior.

**Files:**
- Modify: `integration/config/settings.yaml` (`sensing.loadcells`: enable, port, remap)

**Interfaces:**
- Consumes: everything from Tasks 1–2; `Pipeline._init_loadcells` passes the whole `sensing.loadcells` dict to `LoadCellReader`, so `bin_remap` flows automatically.

- [ ] **Step 1: Review the existing settings.yaml diff first**

`settings.yaml` already had uncommitted edits at session start. Inspect them so the load-cell edit isn't bundled with unrelated changes:

Run: `git diff integration/config/settings.yaml`
Expected: see what's already changed; keep those edits separate in your mind (only the `loadcells` block is this task's concern).

- [ ] **Step 2: Edit the loadcells block**

In `integration/config/settings.yaml`, under `sensing.loadcells`, set:

```yaml
  loadcells:
    enabled: true
    port: "COM5"           # Silicon Labs CP210x (ESP32) — driver installed 2026-06-17
    baudrate: 115200
    stale_after: 5.0
    bin_remap:
      bin_0_0: bin_0_5     # firmware emits bin_0_0; this load cell is physically bin_0_5
```

- [ ] **Step 3: Confirm the raw stream is reachable (port not held by anything else)**

Run (PowerShell, from repo root):

```
python -m pytest integration/tests/test_loadcell_remap.py integration/tests/test_loadcell_count.py -q
```

Expected: PASS (9 passed) — fast regression gate before the live run.

- [ ] **Step 4: Live verification with the real pipeline**

1. Ensure no other process holds COM5 (close any serial monitor).
2. Run: `python -m integration.src.pipeline`
3. Press `1` to calibrate the grid, `2` to init the kit (so `bin_0_5` exists).
4. In the dashboard (http://localhost:8080), find `bin_0_5`:
   - With the bin full, `current` reads `0 / 5` (idle weight ≈ 0 g now that the missing AMP is back).
   - Remove one AMP → within ~1 s `current` ticks to `1 / 5`; remove another → `2 / 5`.
   - Replace an AMP → count drops back. Weight (grams) shows live.
5. Sanity: unplug/replug or stop the ESP32 briefly → the count holds its last value (does not jump to 0) while disconnected.

Expected: `bin_0_5` pick count tracks AMP removals automatically; other bins still respond only to manual +/-.

- [ ] **Step 5: Commit the config**

```bash
git add integration/config/settings.yaml
git commit -m "feat(v2): enable bin_0_5 load cell on COM5 with bin_0_0->bin_0_5 remap"
```

---

## Notes for the implementer

- If `python -m integration.src.pipeline` can't import, run it from the repo root `C:\Users\chenx\Documents\TEnterns\aegis-v2` (the module inserts the needed paths itself).
- `set_pick_count` returns `None` for a bin not yet in state (e.g. before calibration). `_apply_loadcell_counts` ignores the return — harmless; the count applies once `bin_0_5` exists.
- The `amp` unit weight (3.8 g) is the live-confirmed per-unit mass. If counts read consistently high/low by one, re-measure a known stack and adjust `inventory.yaml` only.
