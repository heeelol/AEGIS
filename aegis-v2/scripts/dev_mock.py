"""
AEGIS v2 — Dev Mock for the Operator Dashboard
==============================================
Spins up the FastAPI dashboard with synthetic data so you can preview the UI
without a camera, YOLO model, or MediaPipe install.

Run:
    cd aegis-v2
    py scripts/dev_mock.py

Then open http://localhost:8080
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Put aegis-v2/ on sys.path so `integration.*` imports resolve
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from integration.src.ui.dashboard import start_dashboard
from integration.src.ui.state import PipelineState
from integration.src.detectors.layout_mapper import map_detections_to_layout


# ── Minimal duck-typed stand-ins for the real Hand / BinEvent ──

@dataclass
class FakeHand:
    hand_id: int
    handedness: str
    center: tuple[float, float]
    is_grabbing: bool
    grab_score: float


@dataclass
class FakeEvent:
    hand_id: int
    bin_id: Optional[str]
    handedness: str


# ── Mock world ─────────────────────────────────────────────────

# The real rig: 6 single-slot bins on top, 3 double-slot bins on the bottom.
LAYOUT = [[1, 1, 1, 1, 1, 1], [2, 2, 2]]
FRAME_W, FRAME_H = 1280, 720


def _box(row, slot_start, span, row_slots, num_rows=2):
    """Synthetic detection box centred on a slot's expected position."""
    ex = (slot_start + span / 2.0) / row_slots
    xc = ex * FRAME_W
    half_w = (span / row_slots) * FRAME_W * 0.4
    yc = (row + 0.5) / num_rows * FRAME_H
    half_h = (FRAME_H / num_rows) * 0.3
    return (xc - half_w, yc - half_h, xc + half_w, yc + half_h,
            round(random.uniform(0.85, 0.98), 2))


def build_geofences() -> dict:
    """Run synthetic detections through the LayoutMapper.

    Deliberately omits the bottom-middle bin (slot 2 → bin_1_1) so the
    dashboard shows a grey 'undetected' placeholder in the right slot.
    """
    boxes = [_box(0, c, 1, 6) for c in range(6)]          # full top row
    boxes += [_box(1, 0, 2, 6), _box(1, 4, 2, 6)]         # bottom: skip slot 2
    return map_detections_to_layout(boxes, LAYOUT, FRAME_W, FRAME_H)


# Scripted FSM sequence per pick attempt
FSM_SEQUENCE = [
    ("idle",           0.0, 0.5),
    ("gate_1_spatial", 2.0, 2.0),
    ("gate_2_intent",  1.5, 1.5),
    ("gate_3_verify",  3.0, 3.0),
    ("success",        0.0, 0.8),
]

FSM_ERROR_SEQUENCE = [
    ("idle",           0.0, 0.5),
    ("gate_1_spatial", 2.0, 2.0),
    ("gate_2_intent",  1.5, 1.5),
    ("error",          0.0, 1.5),
]


def bin_center(geofences: dict, bin_id: str) -> tuple[float, float]:
    g = geofences[bin_id]
    return ((g["x_min"] + g["x_max"]) / 2.0, (g["y_min"] + g["y_max"]) / 2.0)


def jitter(point: tuple[float, float], amount: float = 8.0) -> tuple[float, float]:
    return (point[0] + random.uniform(-amount, amount),
            point[1] + random.uniform(-amount, amount))


# ── Main mock loop ─────────────────────────────────────────────

def main() -> None:
    state = PipelineState()
    geofences = build_geofences()

    # Initial bin map push
    state.update_bins(geofences)

    # Work order: targets for some bins, explicit zeros for the rest so
    # they render grey (and flip to "wrong_bin" red when a hand enters them)
    work_order = {bid: 0 for bid in geofences}
    work_order.update({
        "bin_0_0": 3,
        "bin_0_1": 5,
        "bin_0_2": 2,
        "bin_0_3": 4,
        "bin_1_0": 3,
        "bin_1_2": 2,
    })
    state.set_work_order(work_order)

    # Start the FastAPI dashboard in a background thread
    start_dashboard(state, port=8080)
    print("\n  AEGIS v2 mock dashboard:  http://localhost:8080\n")
    print("  Press Ctrl+C to stop.\n")

    target_bins = ["bin_0_0", "bin_0_1", "bin_0_2", "bin_0_3", "bin_1_0", "bin_1_2"]
    wrong_bins = ["bin_0_4", "bin_0_5"]   # detected but not in the job → red flash
    frame = 0
    t0 = time.time()

    while True:
        # Pick a bin to interact with this cycle
        # ~20% of the time pick a wrong bin (fires the wrong_bin status + error)
        will_error = random.random() < 0.25
        if will_error and random.random() < 0.5:
            target = random.choice(wrong_bins)
            sequence = FSM_ERROR_SEQUENCE
        else:
            target = random.choice(target_bins)
            sequence = FSM_ERROR_SEQUENCE if will_error else FSM_SEQUENCE

        target_center = bin_center(geofences, target)

        # Other hand wanders away from the target
        other_center = (random.uniform(100, 1180), random.uniform(100, 620))

        # Run through the scripted FSM sequence
        for fsm_state, total_time, real_duration in sequence:
            sub_steps = max(1, int(real_duration * 5))  # 5 Hz updates
            for s in range(sub_steps):
                elapsed_in_state = (s / sub_steps) * total_time

                hand_grab = fsm_state in ("gate_2_intent", "gate_3_verify")
                grab_score = 0.85 if hand_grab else random.uniform(0.05, 0.25)

                hands = [
                    FakeHand(
                        hand_id=0,
                        handedness="Right",
                        center=jitter(target_center),
                        is_grabbing=hand_grab,
                        grab_score=grab_score,
                    ),
                    FakeHand(
                        hand_id=1,
                        handedness="Left",
                        center=jitter(other_center, 30),
                        is_grabbing=False,
                        grab_score=random.uniform(0.02, 0.15),
                    ),
                ]
                events = [
                    FakeEvent(hand_id=0, bin_id=target, handedness="Right"),
                    FakeEvent(hand_id=1, bin_id=None, handedness="Left"),
                ]

                state.update_hands(hands, events)
                state.update_fsm(
                    state=fsm_state,
                    bin_id=target if fsm_state != "idle" else None,
                    elapsed=elapsed_in_state,
                )

                # On success, increment the bin's pick count
                if fsm_state == "success" and s == 0:
                    state.record_pick(target)

                # On error, add to error overlay feed
                if fsm_state == "error" and s == 0:
                    reasons = [
                        "Wrong bin reached",
                        "Hand released too early",
                        "Pick verification failed",
                    ]
                    state.add_error(target, random.choice(reasons))

                # FPS / frame counter
                frame += 1
                elapsed_total = max(time.time() - t0, 1e-6)
                state.update_fps(frame / elapsed_total)

                time.sleep(real_duration / sub_steps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMock stopped.")
