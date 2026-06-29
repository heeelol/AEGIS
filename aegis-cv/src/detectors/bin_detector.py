"""
Dynamic Bin Detection using YOLOv8
Automatically detects bin locations and generates geofence coordinates in real-time.
Eliminates need for manual calibration of bins_map.yaml.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
import cv2

logger = logging.getLogger(__name__)


class BinDetector:
    """Dynamic bin detection and geofence coordinate extraction."""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.6):
        """
        Initialize bin detector with YOLOv8.
        
        Args:
            model_path: Path to YOLOv8 model (pretrained or custom)
            conf_threshold: Detection confidence threshold (0-1)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.detected_bins = {}
        self.bin_order_cache = None  # Cache for consistent bin ordering
        logger.info(f"Loaded bin detector model: {model_path}")
    
    def detect_bins(self, frame: np.ndarray) -> Dict[str, Dict]:
        """
        Detect all bins in a frame and extract geofence coordinates.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            Dictionary mapping bin_id to geofence coordinates
            {
                'bin_0_0': {'x_min': 10, 'x_max': 120, 'y_min': 10, 'y_max': 120},
                'bin_0_1': {...},
                ...
            }
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        if not results or len(results) == 0:
            logger.warning("No bins detected in frame")
            return {}
        
        detections = results[0]
        
        if len(detections.boxes) == 0:
            logger.warning("No bin bounding boxes found")
            return {}
        
        # Extract bounding boxes
        boxes = detections.boxes.xyxy.cpu().numpy()  # (x_min, y_min, x_max, y_max)
        confidences = detections.boxes.conf.cpu().numpy()
        
        # Sort boxes by position (left-to-right, top-to-bottom)
        sorted_boxes, sorted_confs = self._sort_boxes_spatially(boxes, confidences)
        
        # Assign bin IDs based on spatial position
        geofences = self._assign_bin_ids(sorted_boxes, sorted_confs)
        
        self.detected_bins = geofences
        return geofences
    
    def _sort_boxes_spatially(
        self, boxes: np.ndarray, confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort bounding boxes spatially (row-major order: left-to-right, top-to-bottom).
        
        Args:
            boxes: Array of bounding boxes (N, 4)
            confidences: Detection confidence scores (N,)
            
        Returns:
            Sorted boxes and confidences
        """
        if len(boxes) == 0:
            return boxes, confidences
        
        # Extract centers
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        
        # Sort by y (row), then x (column)
        y_threshold = np.std(centers_y) * 0.5  # Threshold for grouping rows
        
        indices = []
        sorted_y = np.sort(np.unique(centers_y))
        
        for y_val in sorted_y:
            row_mask = np.abs(centers_y - y_val) < y_threshold
            row_indices = np.where(row_mask)[0]
            row_indices = row_indices[np.argsort(centers_x[row_indices])]
            indices.extend(row_indices)
        
        indices = np.array(indices)
        return boxes[indices], confidences[indices]
    
    def _assign_bin_ids(self, boxes: np.ndarray, confidences: np.ndarray) -> Dict[str, Dict]:
        """
        Assign bin IDs to detected boxes and extract geofence coordinates.
        Assumes 4x2 array (8 bins total): 4 columns × 2 rows.
        
        Args:
            boxes: Sorted bounding boxes
            confidences: Detection confidences
            
        Returns:
            Dictionary of geofence coordinates per bin
        """
        geofences = {}
        
        # Expected: 8 bins (4 columns × 2 rows)
        if len(boxes) != 8:
            logger.warning(f"Expected 8 bins, detected {len(boxes)}. Proceeding anyway...")
        
        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            x_min, y_min, x_max, y_max = box
            
            # Calculate bin position in grid (4 columns × 2 rows)
            row = idx // 4  # 0 or 1
            col = idx % 4   # 0, 1, 2, or 3
            
            bin_id = f"bin_{row}_{col}"
            
            # Add margin for robustness
            margin = 3  # pixels
            
            geofences[bin_id] = {
                'x_min': max(0, int(x_min) - margin),
                'x_max': int(x_max) + margin,
                'y_min': max(0, int(y_min) - margin),
                'y_max': int(y_max) + margin,
                'confidence': float(conf),
            }
            
            logger.debug(
                f"Detected {bin_id} at ({int(x_min)}, {int(y_min)}) "
                f"- ({int(x_max)}, {int(y_max)}) [conf: {conf:.2f}]"
            )
        
        return geofences
    
    def visualize_detections(
        self, frame: np.ndarray, geofences: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Visualize detected bins on frame with bounding boxes and labels.
        
        Args:
            frame: Input image
            geofences: Geofence dictionary (uses self.detected_bins if None)
            
        Returns:
            Frame with drawn bounding boxes and labels
        """
        if geofences is None:
            geofences = self.detected_bins
        
        viz_frame = frame.copy()
        
        for bin_id, coords in geofences.items():
            x_min = coords['x_min']
            y_min = coords['y_min']
            x_max = coords['x_max']
            y_max = coords['y_max']
            confidence = coords.get('confidence', 0.0)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(viz_frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label
            label = f"{bin_id} ({confidence:.2f})"
            cv2.putText(
                viz_frame,
                label,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        
        return viz_frame
    
    def get_geofences(self) -> Dict[str, Dict]:
        """Get most recent detected geofences."""
        return self.detected_bins


class DynamicGeofenceManager:
    """
    Manages dynamic geofences from continuous bin detection.
    Replaces static bins_map.yaml with real-time detection.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", smoothing_window: int = 5):
        """
        Initialize dynamic geofence manager.
        
        Args:
            model_path: Path to YOLOv8 model
            smoothing_window: Number of frames for coordinate smoothing
        """
        self.detector = BinDetector(model_path)
        self.smoothing_window = smoothing_window
        self.history = []  # History of geofence detections for smoothing
    
    def update(self, frame: np.ndarray) -> Dict[str, Dict]:
        """
        Update geofences based on new frame.
        Applies temporal smoothing for stability.
        
        Args:
            frame: Input image frame
            
        Returns:
            Smoothed geofence coordinates
        """
        # Detect bins in current frame
        geofences = self.detector.detect_bins(frame)
        
        if not geofences:
            # Use previous detection if current fails
            if self.history:
                return self.history[-1]
            return {}
        
        # Add to history
        self.history.append(geofences)
        if len(self.history) > self.smoothing_window:
            self.history.pop(0)
        
        # Apply temporal smoothing
        smoothed = self._smooth_geofences(self.history)
        
        return smoothed
    
    def _smooth_geofences(self, history: List[Dict]) -> Dict[str, Dict]:
        """
        Smooth geofence coordinates over time for stability.
        Uses median of recent detections.
        
        Args:
            history: List of geofence detections over time
            
        Returns:
            Smoothed geofence coordinates
        """
        if not history:
            return {}
        
        # Get all bin IDs from latest detection
        latest = history[-1]
        smoothed = {}
        
        for bin_id in latest.keys():
            coords_x_min = []
            coords_x_max = []
            coords_y_min = []
            coords_y_max = []
            
            # Collect coordinates from history
            for geofence_dict in history:
                if bin_id in geofence_dict:
                    coords = geofence_dict[bin_id]
                    coords_x_min.append(coords['x_min'])
                    coords_x_max.append(coords['x_max'])
                    coords_y_min.append(coords['y_min'])
                    coords_y_max.append(coords['y_max'])
            
            # Use median for smoothing (robust to outliers)
            if coords_x_min:
                smoothed[bin_id] = {
                    'x_min': int(np.median(coords_x_min)),
                    'x_max': int(np.median(coords_x_max)),
                    'y_min': int(np.median(coords_y_min)),
                    'y_max': int(np.median(coords_y_max)),
                    'confidence': latest[bin_id].get('confidence', 0.0),
                }
        
        return smoothed
    
    def check_hand_in_geofence(
        self, hand_keypoints: List[Tuple[float, float]], geofences: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if hand keypoints are within detected geofences.
        
        Args:
            hand_keypoints: List of (x, y) coordinates
            geofences: Dictionary of geofence coordinates
            
        Returns:
            (is_in_geofence, bin_id)
        """
        points_in_geofence = []
        
        for x, y in hand_keypoints:
            if x is None or y is None:
                continue
            
            for bin_id, coords in geofences.items():
                if (coords['x_min'] <= x <= coords['x_max'] and
                    coords['y_min'] <= y <= coords['y_max']):
                    points_in_geofence.append(bin_id)
                    break
        
        # Require at least 2 keypoints for robustness
        if len(points_in_geofence) >= 2:
            bin_id = points_in_geofence[0]
            return True, bin_id
        
        return False, None
