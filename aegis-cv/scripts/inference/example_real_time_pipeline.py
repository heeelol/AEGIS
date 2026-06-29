"""
Example: Integration of Dynamic Bin Detection with FSM
Shows how to use DynamicGeofenceManager with TripleGateFSM for real-time procedural verification.
"""

import logging
import cv2
import time
from pathlib import Path

from src.detectors.bin_detector import DynamicGeofenceManager
# from src.detectors.pose_estimator import PoseEstimator  # (to be implemented)
from src.engine.fsm import TripleGateFSM, SensorReading
from src.sensing.modbus_client import ModbusWeightClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeVerificationPipeline:
    """
    Real-time procedural verification pipeline combining:
    - Dynamic bin detection
    - Hand pose estimation
    - Triple-Gate FSM verification
    - Weight confirmation
    """
    
    def __init__(self, config: dict):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary from settings.yaml
        """
        self.config = config
        
        # Initialize components
        self.geofence_mgr = DynamicGeofenceManager(
            model_path="models/pretrained/yolov8n.pt",  # Bin detector model
            smoothing_window=5
        )
        
        # TODO: Implement PoseEstimator for hand keypoints
        # self.pose_estimator = PoseEstimator("models/custom/grab_pose.engine")
        
        # Initialize FSM
        self.fsm = TripleGateFSM(config)
        
        # TODO: Initialize Modbus client for weight sensing
        # self.modbus_client = ModbusWeightClient(config['sensing']['modbus'])
        
        logger.info("Initialized Real-Time Verification Pipeline")
    
    def run(self, video_source: int = 0):
        """
        Main loop: Sense-Analyse-Act.
        
        Args:
            video_source: Camera device ID or video file path
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {video_source}")
            return
        
        frame_count = 0
        start_time = time.time()
        previous_weight = {}
        
        logger.info("Starting Sense-Analyse-Act loop...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # ========== SENSE ==========
            # 1. Detect bins dynamically
            geofences = self.geofence_mgr.update(frame)
            
            # 2. Estimate hand pose
            # hand_keypoints = self.pose_estimator.estimate(frame)
            hand_keypoints = []  # Placeholder
            
            # 3. Check if hand in geofence
            hand_in_geofence, bin_id = self.geofence_mgr.check_hand_in_geofence(
                hand_keypoints, geofences
            )
            
            # 4. Read weight from Modbus
            # weight_data = await self.modbus_client.read_all_weights()
            weight_data = {}  # Placeholder
            
            # Calculate weight delta
            weight_delta = 0.0
            if bin_id and bin_id in previous_weight:
                weight_delta = abs(weight_data.get(bin_id, 0) - previous_weight[bin_id])
            
            # ========== ANALYSE ==========
            # Run FSM with sensor readings
            sensor_reading = SensorReading(
                timestamp=time.time(),
                hand_in_geofence=hand_in_geofence,
                closed_fist_detected=False,  # TODO: Detect from pose
                weight_delta=weight_delta,
                bin_id=bin_id
            )
            
            fsm_state = self.fsm.update(sensor_reading)
            
            # ========== ACT ==========
            if fsm_state.value == "success":
                logger.info(f"✓✓✓ PICK VERIFIED: {bin_id}")
                # TODO: Update inventory database
                # TODO: Publish event to UI
            
            elif fsm_state.value == "error":
                logger.warning(f"✗ PICK FAILED: Check {bin_id}")
                # TODO: Trigger visual alert on dashboard
            
            # Update weight cache
            if weight_data:
                previous_weight = weight_data.copy()
            
            # ========== VISUALIZATION ==========
            # Draw geofences and FSM state
            frame_viz = self._visualize_state(frame, geofences, fsm_state, bin_id)
            
            cv2.imshow("AEGIS Real-Time Verification", frame_viz)
            
            # Calculate latency
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                latency_ms = (elapsed / frame_count) * 1000
                logger.info(f"FPS: {fps:.1f}, Latency: {latency_ms:.1f}ms")
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Processed {frame_count} frames")
    
    def _visualize_state(self, frame, geofences, fsm_state, active_bin_id):
        """Visualize geofences, hand detection, and FSM state."""
        viz_frame = frame.copy()
        
        # Draw geofences
        for bin_id, coords in geofences.items():
            x_min, y_min = coords['x_min'], coords['y_min']
            x_max, y_max = coords['x_max'], coords['y_max']
            
            # Highlight active bin
            color = (0, 255, 0) if bin_id == active_bin_id else (200, 200, 200)
            
            cv2.rectangle(viz_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(
                viz_frame, bin_id, (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        # Draw FSM state
        state_color = {
            'idle': (200, 200, 200),
            'gate_1_spatial': (255, 165, 0),
            'gate_2_intent': (255, 100, 0),
            'gate_3_verify': (0, 165, 255),
            'success': (0, 255, 0),
            'error': (0, 0, 255),
        }
        
        color = state_color.get(fsm_state.value, (255, 255, 255))
        cv2.putText(
            viz_frame,
            f"FSM: {fsm_state.value.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        
        return viz_frame


if __name__ == "__main__":
    # Load config
    import yaml
    
    config_path = Path("config/settings.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run pipeline
    pipeline = RealTimeVerificationPipeline(config)
    pipeline.run(video_source=0)  # Use default camera
