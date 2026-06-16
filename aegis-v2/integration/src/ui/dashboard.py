"""
AEGIS v2 — FastAPI Dashboard Backend
======================================
Serves the operator web dashboard and provides real-time API endpoints.

The dashboard shows:
  - Bin grid with color-coded status (white/orange/green/red)
  - Hand tracking positions and grab status
  - FSM gate progress
  - Error alerts with full-screen overlay
  - Live FPS and system stats

The pipeline pushes state into a shared PipelineState object;
the API endpoints read from it. No direct coupling to CV code.

Usage:
    # Standalone (for development)
    uvicorn integration.src.ui.dashboard:app --reload --port 8080

    # From pipeline (auto-launched in background thread)
    from integration.src.ui.dashboard import start_dashboard
    start_dashboard(state, port=8080)
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .state import PipelineState

logger = logging.getLogger("aegis.ui.dashboard")

# Resolve static files directory
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Shared state — set by start_dashboard() or directly
_state: Optional[PipelineState] = None


def create_app(state: Optional[PipelineState] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _state
    if state is not None:
        _state = state

    app = FastAPI(
        title="AEGIS v2 Dashboard",
        description="Real-time operator dashboard for bin tracking & kinetic gating",
        version="2.0.0",
    )

    # Allow local dev connections
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API Endpoints ────────────────────────────────────────

    @app.get("/api/bins")
    def get_bins():
        """Return all bins with calculated status."""
        if _state is None:
            return JSONResponse([], status_code=200)
        return _state.get_bins()

    @app.get("/api/layout")
    def get_layout():
        """Return the bin grid layout (layers × bins) merged from CV + load cells."""
        if _state is None:
            return {"num_layers": 0, "num_bins": 0, "layers": [],
                    "source": {"cv": False, "loadcells": False}}
        return _state.get_layout()

    @app.get("/api/hands")
    def get_hands():
        """Return current hand detection data."""
        if _state is None:
            return []
        return _state.get_hands()

    @app.get("/api/fsm")
    def get_fsm():
        """Return current FSM state."""
        if _state is None:
            return {"state": "idle", "bin_id": None, "elapsed": 0}
        return _state.get_fsm()

    @app.get("/api/errors")
    def get_errors():
        """Return recent error events."""
        if _state is None:
            return []
        return _state.get_errors()

    @app.get("/api/stats")
    def get_stats():
        """Return system stats (FPS, uptime, frame count)."""
        if _state is None:
            return {"fps": 0, "frame_count": 0, "uptime_seconds": 0}
        return _state.get_stats()

    @app.post("/api/errors/clear")
    def clear_errors():
        """Clear all error events."""
        if _state:
            _state.clear_errors()
        return {"status": "ok"}

    @app.post("/api/work-order")
    def set_work_order(targets: dict):
        """
        Set target pick counts per bin.
        Body: {"bin_0_0": 5, "bin_0_1": 3, ...}
        """
        if _state:
            _state.set_work_order(targets)
        return {"status": "ok"}

    @app.post("/api/bins/{bin_id}/pick")
    def override_pick_count(bin_id: str, body: dict):
        """
        Manual override for the load-cell pick count on a bin.
        Body: {"delta": ±N}  -- nudge by N (signed)
              {"count": N}   -- set to absolute N
        Returns the new pick count, or 404 if the bin is unknown.
        """
        if _state is None:
            return JSONResponse({"error": "no state"}, status_code=503)

        if "delta" in body:
            new_val = _state.adjust_pick_count(bin_id, body["delta"])
        elif "count" in body:
            new_val = _state.set_pick_count(bin_id, body["count"])
        else:
            return JSONResponse(
                {"error": "body must include 'delta' or 'count'"},
                status_code=400,
            )

        if new_val is None:
            return JSONResponse({"error": f"unknown bin {bin_id}"}, status_code=404)
        return {"bin_id": bin_id, "current": new_val}

    # ── Static files & index ─────────────────────────────────

    @app.get("/")
    def index():
        return FileResponse(_STATIC_DIR / "index.html")

    # Mount static files (CSS, JS)
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


# Default app instance (for uvicorn standalone usage)
app = create_app()


def start_dashboard(
    state: PipelineState,
    host: str = "0.0.0.0",
    port: int = 8080,
    open_browser: bool = True,
) -> threading.Thread:
    """
    Launch the dashboard in a background daemon thread.

    Called by the pipeline to start the web UI alongside the CV loop.
    When ``open_browser`` is set, the operator UI is opened in the default
    browser shortly after the server starts. Returns the thread handle.
    """
    import uvicorn

    global _state
    _state = state

    _app = create_app(state)

    def _run():
        logger.info("Dashboard starting on http://%s:%d", host, port)
        uvicorn.run(_app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True, name="aegis-dashboard")
    thread.start()
    logger.info("Dashboard thread started")

    if open_browser:
        # Open the operator UI once the server has had a moment to bind. 0.0.0.0
        # isn't browsable, so always point the browser at localhost. Runs on a
        # daemon Timer so a headless/no-browser host never blocks or hangs exit.
        url = f"http://localhost:{port}"

        def _open():
            try:
                webbrowser.open(url)
            except Exception as e:  # pragma: no cover - environment dependent
                logger.warning("Could not open browser at %s: %s", url, e)

        opener = threading.Timer(1.0, _open)
        opener.daemon = True
        opener.start()

    return thread
