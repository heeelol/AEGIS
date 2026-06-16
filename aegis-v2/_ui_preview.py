"""
Throwaway preview server for the kitting operator console.

Simplified model:
  - each bin tracks `removed` (net taken out of the bin) and `placed`
    (how many of its items are in the kit box).
  - a bin is DONE (green) only when placed == target AND removed == target
    (the box has its items AND no extras are still in hand).
  - the kit box is not shown; its sensor only guards faults (overpack etc.).

Three per-bin actions drive the flow (a load-cell / vision stand-in):
  grab   : take one from the bin            removed += 1
  place  : move one from hand into the box  placed  += 1   (needs hand stock)
  return : put one from hand back in bin    removed -= 1   (needs hand stock)

Delete when done.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).parent / "integration" / "src" / "ui" / "static"
app = FastAPI()

BINS = {
    "bin_0_0": {"label": "A1", "in_bom": True,  "detected": True,  "target": 5, "removed": 3, "placed": 3, "hand": False, "handedness": ""},
    "bin_0_1": {"label": "A2", "in_bom": True,  "detected": True,  "target": 5, "removed": 5, "placed": 5, "hand": False, "handedness": ""},
    "bin_0_2": {"label": "A3", "in_bom": True,  "detected": False, "target": 2, "removed": 0, "placed": 0, "hand": False, "handedness": ""},
    "bin_1_0": {"label": "B1", "in_bom": True,  "detected": True,  "target": 4, "removed": 0, "placed": 0, "hand": False, "handedness": ""},
    "bin_1_1": {"label": "B2", "in_bom": False, "detected": True,  "target": 0, "removed": 0, "placed": 0, "hand": False, "handedness": ""},
    "bin_1_2": {"label": "B3", "in_bom": True,  "detected": True,  "target": 3, "removed": 2, "placed": 2, "hand": True,  "handedness": "left"},
}

KIT = {"name": "Starter bundle", "batch": {"done": 45, "target": 200}}

ALERT = {"active": False, "message": ""}
ALERT_MESSAGES = {
    "pick_wrong":   "WRONG BIN — RETURN ITEM TO BIN B2",
    "return_wrong": "WRONG BIN — REMOVE ITEM FROM BIN B2",
    "remove_kit":   "ITEM REMOVED — REPLACE IN KIT BOX",
    "overpack_kit": "KIT OVERPACKED — REMOVE 1 ITEM FROM KIT BOX",
}


# ── Reads ────────────────────────────────────────────

@app.get("/api/bins")
def bins():
    return [{"id": bid, **b} for bid, b in BINS.items()]


@app.get("/api/layout")
def layout():
    return {
        "num_layers": 2, "num_bins": len(BINS),
        "source": {"cv": True, "loadcells": False},
        "layers": [
            {"layer": 0, "row_slots": 3, "num_bins": 3, "bins": [
                {"id": f"bin_0_{c}", "slot_start": c, "span": 1, "detected": True} for c in range(3)]},
            {"layer": 1, "row_slots": 3, "num_bins": 3, "bins": [
                {"id": f"bin_1_{c}", "slot_start": c, "span": 1, "detected": True} for c in range(3)]},
        ],
    }


@app.get("/api/kit")
def kit():
    """Box sensor stays in the backend (fault guard); only batch is shown."""
    live = [b for b in BINS.values() if b["in_bom"] and b["detected"] and b["target"] > 0]
    return {
        "name": KIT["name"],
        "placed": sum(b["placed"] for b in live),
        "total": sum(b["target"] for b in live),
        "in_hand": sum(b["removed"] - b["placed"] for b in BINS.values() if b["detected"]),
        "batch": KIT["batch"],
    }


@app.get("/api/alert")
def get_alert():
    return ALERT


@app.get("/api/stats")
def stats():
    return {"fps": 30, "frame_count": 18432, "uptime_seconds": 754,
            "num_bins": len(BINS), "num_hands": sum(b["hand"] for b in BINS.values()),
            "mock": True}


# ── Per-bin actions ──────────────────────────────────

@app.post("/api/bins/{bin_id}/grab")
def grab(bin_id: str):
    b = BINS.get(bin_id)
    if b is None:
        return JSONResponse({"error": "unknown bin"}, status_code=404)
    b["removed"] += 1
    return {"id": bin_id, "removed": b["removed"], "placed": b["placed"]}


@app.post("/api/bins/{bin_id}/place")
def place(bin_id: str):
    b = BINS.get(bin_id)
    if b is None:
        return JSONResponse({"error": "unknown bin"}, status_code=404)
    if b["removed"] - b["placed"] > 0:
        b["placed"] += 1
    return {"id": bin_id, "removed": b["removed"], "placed": b["placed"]}


@app.post("/api/bins/{bin_id}/return")
def return_to_bin(bin_id: str):
    b = BINS.get(bin_id)
    if b is None:
        return JSONResponse({"error": "unknown bin"}, status_code=404)
    if b["removed"] - b["placed"] > 0:
        b["removed"] -= 1
    return {"id": bin_id, "removed": b["removed"], "placed": b["placed"]}


# ── Kit completion ───────────────────────────────────

@app.post("/api/kit/complete")
def kit_complete():
    KIT["batch"]["done"] += 1
    for b in BINS.values():
        if b["in_bom"] and b["detected"]:
            b["removed"] = 0
            b["placed"] = 0
    return {"status": "ok", "batch": KIT["batch"]}


# ── Mock-only controls ───────────────────────────────

@app.post("/api/mock/hand")
def mock_hand(body: dict):
    b = BINS.get(body.get("bin"))
    if b is None:
        return JSONResponse({"error": "unknown bin"}, status_code=404)
    b["hand"] = bool(body.get("on", True))
    b["handedness"] = body.get("handedness", "right") if b["hand"] else ""
    return {"bin": body.get("bin"), "hand": b["hand"]}


@app.post("/api/mock/alert")
def set_alert(body: dict):
    msg = ALERT_MESSAGES.get(body.get("scenario"))
    if not msg:
        return JSONResponse({"error": "unknown scenario"}, status_code=400)
    ALERT["active"] = True
    ALERT["message"] = msg
    return ALERT


@app.post("/api/mock/reset-faults")
def mock_reset_faults():
    """Clear hands, clear the alert, and return everything in hand to its bin."""
    for b in BINS.values():
        b["hand"] = False
        b["handedness"] = ""
        b["removed"] = b["placed"]
    ALERT["active"] = False
    ALERT["message"] = ""
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
