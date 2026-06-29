"""
Throwaway preview server that mimics the REAL dashboard.py contract
(integration/src/ui/state.py get_bins / get_layout), so the kitting UI can be
previewed without the CV stack (cv2/torch). Seeded with the real 6+3 rig:
6 top single-slot bins + 3 bottom 2-slot bins, including one missing bin and a
hand in a bin. Delete when done.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# this file lives in aegis-v2/tools/, so the UI static dir is one level up
STATIC = Path(__file__).parent.parent / "integration" / "src" / "ui" / "static"
app = FastAPI()

# Real rig: 6 top (1 slot each) + 3 bottom (2 slots each), 6-slot track per row.
BINS = {
    "bin_0_0": {"layer": 0, "col": 0, "slot_start": 0, "span": 1, "target": 5, "current": 3, "detected": True,  "is_active": True,  "handedness": "left", "weight": 142},
    "bin_0_1": {"layer": 0, "col": 1, "slot_start": 1, "span": 1, "target": 0, "current": 0, "detected": True,  "is_active": False, "handedness": "", "weight": 0},
    "bin_0_2": {"layer": 0, "col": 2, "slot_start": 2, "span": 1, "target": 5, "current": 5, "detected": True,  "is_active": False, "handedness": "", "weight": 240},
    "bin_0_3": {"layer": 0, "col": 3, "slot_start": 3, "span": 1, "target": 5, "current": 0, "detected": True,  "is_active": True,  "handedness": "right", "weight": 0},
    "bin_0_4": {"layer": 0, "col": 4, "slot_start": 4, "span": 1, "target": 0, "current": 0, "detected": True,  "is_active": False, "handedness": "", "weight": 0},
    "bin_0_5": {"layer": 0, "col": 5, "slot_start": 5, "span": 1, "target": 5, "current": 0, "detected": False, "is_active": False, "handedness": "", "weight": 0},
    "bin_1_0": {"layer": 1, "col": 0, "slot_start": 0, "span": 2, "target": 3, "current": 1, "detected": True,  "is_active": False, "handedness": "", "weight": 90},
    "bin_1_1": {"layer": 1, "col": 1, "slot_start": 2, "span": 2, "target": 3, "current": 3, "detected": True,  "is_active": False, "handedness": "", "weight": 270},
    "bin_1_2": {"layer": 1, "col": 2, "slot_start": 4, "span": 2, "target": 3, "current": 0, "detected": True,  "is_active": False, "handedness": "", "weight": 0},
}
ROW_SLOTS = 6


def _using(b):
    return b["target"] > 0


def _status(b):
    if not b["detected"]:
        return "grey"
    if not _using(b):
        return "wrong_bin" if b["is_active"] else "grey"
    if b["is_active"]:
        return "active"
    if b["target"] == 0 or b["current"] == 0:
        return "white"
    if b["current"] < b["target"]:
        return "orange"
    if b["current"] == b["target"]:
        return "green"
    return "warn"


@app.get("/api/bins")
def get_bins():
    return [
        {"id": bid, "label": bid, "layer": b["layer"], "col": b["col"],
         "current": b["current"], "total": b["target"], "status": _status(b),
         "using": _using(b), "is_active": b["is_active"], "handedness": b["handedness"],
         "confidence": 0.9 if b["detected"] else 0.0, "weight": b["weight"]}
        for bid, b in BINS.items()
    ]


@app.get("/api/layout")
def get_layout():
    layers = []
    for layer in (0, 1):
        row = [b for b in BINS.values() if b["layer"] == layer]
        layers.append({
            "layer": layer, "row_slots": ROW_SLOTS, "num_bins": len(row),
            "bins": [
                {"id": f"bin_{layer}_{b['col']}", "slot_start": b["slot_start"],
                 "span": b["span"], "detected": b["detected"]}
                for b in sorted(row, key=lambda x: x["col"])
            ],
        })
    return {"num_layers": 2, "num_bins": len(BINS), "layers": layers,
            "source": {"cv": True, "loadcells": False}}


@app.get("/api/stats")
def get_stats():
    return {"fps": 30, "frame_count": 18432, "uptime_seconds": 754, "last_update": 0,
            "num_bins": len(BINS), "num_hands": sum(b["is_active"] for b in BINS.values())}


@app.post("/api/bins/{bin_id}/pick")
def override(bin_id: str, body: dict):
    b = BINS.get(bin_id)
    if b is None:
        return JSONResponse({"error": "unknown bin"}, status_code=404)
    if "delta" in body:
        b["current"] = max(0, b["current"] + int(body["delta"]))
    elif "count" in body:
        b["current"] = max(0, int(body["count"]))
    return {"bin_id": bin_id, "current": b["current"]}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
