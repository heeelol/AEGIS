"""
Throwaway preview server for the kitting operator console.
Count-driven model: each bin has a `current` pick count vs its `target`.
The +/- controls adjust `current` (a load-cell stand-in); overpick is
current > target. The full-screen red overlay is driven only by the four
discrete alert events. Delete when done.

Data contract:
  /api/bins  -> [{id,label,in_bom,detected,current,target,hand,handedness}]
  /api/kit   -> {name,placed,total,batch:{done,target}}
  /api/alert -> {active,message}
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).parent / "integration" / "src" / "ui" / "static"
app = FastAPI()

BINS = {
    "bin_0_0": {"label": "A1", "in_bom": True,  "detected": True,  "current": 3, "target": 5, "hand": False, "handedness": ""},
    "bin_0_1": {"label": "A2", "in_bom": True,  "detected": True,  "current": 5, "target": 5, "hand": False, "handedness": ""},
    "bin_0_2": {"label": "A3", "in_bom": True,  "detected": False, "current": 0, "target": 2, "hand": False, "handedness": ""},
    "bin_1_0": {"label": "B1", "in_bom": True,  "detected": True,  "current": 0, "target": 4, "hand": False, "handedness": ""},
    "bin_1_1": {"label": "B2", "in_bom": False, "detected": True,  "current": 0, "target": 0, "hand": False, "handedness": ""},
    "bin_1_2": {"label": "B3", "in_bom": True,  "detected": True,  "current": 2, "target": 3, "hand": True,  "handedness": "left"},
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
    live = [b for b in BINS.values() if b["in_bom"] and b["detected"] and b["target"] > 0]
    return {
        "name": KIT["name"],
        "placed": sum(min(b["current"], b["target"]) for b in live),
        "total": sum(b["target"] for b in live),
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


# ── Pick override (load-cell stand-in) ───────────────

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


# ── Kit completion ───────────────────────────────────

@app.post("/api/kit/complete")
def kit_complete():
    KIT["batch"]["done"] += 1
    for b in BINS.values():
        if b["in_bom"] and b["detected"]:
            b["current"] = 0
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
    """Clear hands, clear the alert, and clamp any overpick back to target."""
    for b in BINS.values():
        b["hand"] = False
        b["handedness"] = ""
        if b["current"] > b["target"]:
            b["current"] = b["target"]
    ALERT["active"] = False
    ALERT["message"] = ""
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
