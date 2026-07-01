"""
Geofence utility for bin boundary checking.
Determines if hand keypoints are within bin coordinates.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BinGeofence:
    """Represents a single bin's geofence boundaries."""
    bin_id: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    margin: float = 5.0  # pixels


class GeofenceManager:
    """Manages multiple bin geofences and boundary checks."""
    
    def __init__(self, bins_config: Dict):
        """
        Initialize geofence manager from config.
        
        Args:
            bins_config: Bins dictionary from bins_map.yaml
        """
        self.geofences = {}
        self.margin = bins_config.get('geofence_margin', 5.0)
        
        # Parse bin coordinates
        for bin_id, coords in bins_config.get('bins', {}).items():
            self.geofences[bin_id] = BinGeofence(
                bin_id=bin_id,
                x_min=coords['x_min'],
                x_max=coords['x_max'],
                y_min=coords['y_min'],
                y_max=coords['y_max'],
                margin=self.margin,
            )
        
        logger.info(f"Loaded {len(self.geofences)} bin geofences")
    
    def check_point_in_geofence(
        self, x: float, y: float, bin_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if point is within any geofence (or specific bin).
        
        Args:
            x: X coordinate
            y: Y coordinate
            bin_id: Specific bin to check (if None, checks all)
            
        Returns:
            (is_in_geofence, bin_id_containing_point)
        """
        if bin_id:
            # Check specific bin
            geofence = self.geofences.get(bin_id)
            if geofence and self._point_in_bounds(x, y, geofence):
                return True, bin_id
            return False, None
        else:
            # Check all bins
            for gf in self.geofences.values():
                if self._point_in_bounds(x, y, gf):
                    return True, gf.bin_id
            return False, None
    
    def check_hand_keypoints_in_geofence(
        self, keypoints: list, bin_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if hand keypoints are within geofence.
        Requires at least 2 keypoints inside for robustness.
        
        Args:
            keypoints: List of (x, y) coordinates
            bin_id: Specific bin to check
            
        Returns:
            (is_in_geofence, bin_id_containing_keypoints)
        """
        if not keypoints:
            return False, None
        
        points_in_geofence = []
        
        for kp in keypoints:
            x, y = kp[0], kp[1]
            if x is None or y is None:
                continue
            
            is_in, detected_bin = self.check_point_in_geofence(x, y, bin_id)
            if is_in:
                points_in_geofence.append(detected_bin)
        
        # Require at least 2 keypoints for robustness
        if len(points_in_geofence) >= 2:
            detected_bin = points_in_geofence[0]
            confidence = len(points_in_geofence) / len(keypoints)
            logger.debug(f"Hand in bin {detected_bin} (confidence: {confidence:.2f})")
            return True, detected_bin
        
        return False, None
    
    def _point_in_bounds(self, x: float, y: float, geofence: BinGeofence) -> bool:
        """Check if point is within geofence with margin."""
        return (
            geofence.x_min - geofence.margin <= x <= geofence.x_max + geofence.margin
            and
            geofence.y_min - geofence.margin <= y <= geofence.y_max + geofence.margin
        )
    
    def get_bin_info(self, bin_id: str) -> Optional[Dict]:
        """Get geofence info for a specific bin."""
        geofence = self.geofences.get(bin_id)
        if geofence:
            return {
                'bin_id': geofence.bin_id,
                'x_min': geofence.x_min,
                'x_max': geofence.x_max,
                'y_min': geofence.y_min,
                'y_max': geofence.y_max,
                'margin': geofence.margin,
            }
        return None
    
    def get_all_bins(self) -> Dict[str, Dict]:
        """Get info for all bins."""
        return {
            bin_id: self.get_bin_info(bin_id)
            for bin_id in self.geofences.keys()
        }
