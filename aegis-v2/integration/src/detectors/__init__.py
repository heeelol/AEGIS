"""Bin detectors for the aegis-v2 pipeline.

Intentionally empty of eager imports: the OBB detector (``initialize_bins_obb``)
and the grid session (``grid_session``) are imported as submodules where needed,
so merely importing the ``detectors`` package does not drag in cv2 / ultralytics.
"""
