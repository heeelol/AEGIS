"""
AEGIS UI — FastAPI Backend
==========================
Serves the operator dashboard and provides API endpoints
for bin status, hand position, and main bin data.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json

# Resolve paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AEGIS Dashboard")


@app.get("/api/bins")
def get_bins():
    """Return bin status with calculated state."""
    with open("../bins.json", "r") as file:
        bin_data = json.load(file)
    with open("../hand_position.json", "r") as file:
        hand_data = json.load(file)

    bins = bin_data["bins"]
    hand_pos = hand_data["hand_over_bin"]

    for bin_item in bins:
        bin_id = bin_item["id"]
        current = int(bin_item["current"])
        total = int(bin_item["total"])
        using = bin_item["using"].lower() == "true"

        if using:
            bin_item["status"] = _calculate_status(current, total)
        else:
            if hand_pos == bin_id:
                bin_item["status"] = "wrong_bin"
            else:
                bin_item["status"] = "grey"

    return bins


@app.get("/api/mainBin")
def get_main_bin():
    """Return main bin configuration."""
    with open(_DATA_DIR / "mainBin.json", "r") as file:
        return json.load(file)


@app.get("/")
def index():
    """Serve the dashboard."""
    return FileResponse(_STATIC_DIR / "index.html")


def _calculate_status(current: int, total: int) -> str:
    """Determine bin fill status color."""
    if current == 0:
        return "white"
    elif current < total:
        return "orange"
    elif current == total:
        return "green"
    else:
        return "warn"


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")