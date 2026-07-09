"""
Foreground Model
================
A fixed-camera hand-presence oracle for the occlusion gate.

The camera and shelf are static, so anything that does not match the learned
background is new content — a hand or arm reaching in. We use an OpenCV MOG2
background subtractor to turn each frame into a binary foreground mask, and
``patch_ratio`` to ask "is there real foreground at this image point?".

The occlusion gate uses this to tell a genuine reach into a TOP bin (real
foreground appears there) from a hand occluded UNDER the shelf (the landmark
model extrapolates a fingertip into the top bin, but no foreground is there).

cv2 lives here, not in BinAssignmentEngine: the engine takes a plain
``presence_fn(px, py) -> float`` closure, so it stays pure and unit-testable.
"""
from __future__ import annotations

import cv2
import numpy as np

# MOG2 labels definite foreground 255, shadows 127, background 0. We keep only
# definite foreground so shadows cast by a reaching arm are not read as a hand.
_FOREGROUND = 255


class ForegroundModel:
    """Wraps a MOG2 background subtractor and exposes a presence oracle."""

    def __init__(
        self,
        patch_size: int = 41,
        warmup_frames: int = 30,
        history: int = 500,
        var_threshold: float = 16.0,
        use_cuda: bool = False,
    ):
        # Optionally run MOG2 on the GPU (cv2.cuda) — frees the CPU and is ~10x
        # faster on a Jetson. Falls back to CPU if CUDA/cv2.cuda isn't available,
        # so the same code runs on the dev laptop. Default False (CPU).
        self._cuda = False
        if use_cuda:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self._bg = cv2.cuda.createBackgroundSubtractorMOG2(
                        history=history, varThreshold=var_threshold, detectShadows=True
                    )
                    self._gpu = cv2.cuda_GpuMat()
                    self._stream = cv2.cuda.Stream()
                    self._cuda = True
            except Exception:
                self._cuda = False
        if not self._cuda:
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=True
            )
        self._patch = int(patch_size)
        self._warmup = int(warmup_frames)
        self._frames = 0

    @property
    def patch_size(self) -> int:
        """Side length (px) of the presence window used by ``patch_ratio``."""
        return self._patch

    @property
    def ready(self) -> bool:
        """True once enough frames have trained the model. Until then the gate
        should fall back to its landmark heuristic rather than trust this."""
        return self._frames >= self._warmup

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Apply the subtractor to ``frame`` and return a binary (0/1) mask."""
        if self._cuda:
            self._gpu.upload(frame)
            gpu_mask = self._bg.apply(self._gpu, -1.0, self._stream)
            self._stream.waitForCompletion()
            raw = gpu_mask.download()
        else:
            raw = self._bg.apply(frame)
        mask = (raw == _FOREGROUND).astype(np.uint8)
        self._frames += 1
        return mask

    def patch_ratio(self, mask: np.ndarray, px: float, py: float,
                    size: int | None = None) -> float:
        """Fraction of foreground in an N×N window of ``mask`` around (px, py).

        The window is clipped to the frame at the edges. Returns 0.0 for an
        out-of-frame point or an empty window.
        """
        size = self._patch if size is None else int(size)
        h, w = mask.shape[:2]
        half = size // 2
        xi, yi = int(round(px)), int(round(py))
        x0, x1 = max(0, xi - half), min(w, xi + half + 1)
        y0, y1 = max(0, yi - half), min(h, yi + half + 1)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        patch = mask[y0:y1, x0:x1]
        return float(patch.mean())

    def region_ratio(self, mask: np.ndarray, x_min: float, y_min: float,
                     x_max: float, y_max: float) -> float:
        """Fraction of foreground inside an axis-aligned box of ``mask``.

        Used by the occlusion hold to ask "is a forearm still in this bottom
        bin?" — clipped to the frame; 0.0 for a degenerate / out-of-frame box.
        """
        h, w = mask.shape[:2]
        x0, x1 = max(0, int(x_min)), min(w, int(x_max))
        y0, y1 = max(0, int(y_min)), min(h, int(y_max))
        if x0 >= x1 or y0 >= y1:
            return 0.0
        return float(mask[y0:y1, x0:x1].mean())
