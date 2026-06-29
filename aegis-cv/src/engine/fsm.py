"""
Triple-Gate FSM (Finite State Machine) Implementation
Implements the Sense-Analyse-Act loop for procedural verification.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
import time
import logging

logger = logging.getLogger(__name__)


class FSMState(Enum):
    """FSM State Definitions."""
    IDLE = "idle"                       # Waiting for activity
    GATE_1_SPATIAL = "gate_1_spatial"   # Hand in geofence
    GATE_2_INTENT = "gate_2_intent"     # Closed fist detected
    GATE_3_VERIFY = "gate_3_verify"     # Weight change confirmed
    SUCCESS = "success"                 # All gates passed
    ERROR = "error"                     # Gate failed


@dataclass
class SensorReading:
    """Unified sensor data structure."""
    timestamp: float
    hand_in_geofence: bool              # Gate 1: Spatial
    closed_fist_detected: bool          # Gate 2: Intent
    weight_delta: float                 # Gate 3: Verification (grams)
    bin_id: Optional[str] = None


class TripleGateFSM:
    """Triple-Gate Verification State Machine."""
    
    def __init__(self, config: dict):
        """
        Initialize FSM with configuration.
        
        Args:
            config: FSM settings from settings.yaml
        """
        self.config = config
        self.state = FSMState.IDLE
        self.previous_state = None
        self.gate_entry_time = None
        self.last_bin_id = None
    
    def update(self, sensor_reading: SensorReading) -> FSMState:
        """
        Update FSM state based on sensor readings.
        
        Args:
            sensor_reading: Current sensor data
            
        Returns:
            Current FSM state after update
        """
        self.previous_state = self.state
        current_time = time.time()
        
        # State machine transitions
        if self.state == FSMState.IDLE:
            if sensor_reading.hand_in_geofence:
                self._enter_gate_1(sensor_reading, current_time)
            
        elif self.state == FSMState.GATE_1_SPATIAL:
            if self._check_gate_1_timeout(current_time):
                self._fail_gate(sensor_reading)
            elif sensor_reading.closed_fist_detected:
                self._enter_gate_2(sensor_reading, current_time)
            elif not sensor_reading.hand_in_geofence:
                self._reset_to_idle()
        
        elif self.state == FSMState.GATE_2_INTENT:
            if self._check_gate_2_timeout(current_time):
                self._fail_gate(sensor_reading)
            elif sensor_reading.weight_delta >= self.config['fsm']['weight_delta_threshold']:
                self._enter_gate_3(sensor_reading, current_time)
        
        elif self.state == FSMState.GATE_3_VERIFY:
            if self._check_gate_3_timeout(current_time):
                self._fail_gate(sensor_reading)
            else:
                self._success_gate(sensor_reading)
        
        elif self.state == FSMState.SUCCESS:
            self._reset_to_idle()
        
        elif self.state == FSMState.ERROR:
            if current_time - self.gate_entry_time >= self.config['fsm']['error_cooldown']:
                self._reset_to_idle()
        
        return self.state
    
    def _enter_gate_1(self, reading: SensorReading, current_time: float) -> None:
        """Enter Gate 1 (Spatial)."""
        self.state = FSMState.GATE_1_SPATIAL
        self.gate_entry_time = current_time
        self.last_bin_id = reading.bin_id
        logger.info(f"✓ Gate 1 SPATIAL triggered (Bin: {reading.bin_id})")
    
    def _enter_gate_2(self, reading: SensorReading, current_time: float) -> None:
        """Enter Gate 2 (Intent)."""
        self.state = FSMState.GATE_2_INTENT
        self.gate_entry_time = current_time
        logger.info(f"✓ Gate 2 INTENT triggered (Closed Fist detected)")
    
    def _enter_gate_3(self, reading: SensorReading, current_time: float) -> None:
        """Enter Gate 3 (Verification)."""
        self.state = FSMState.GATE_3_VERIFY
        self.gate_entry_time = current_time
        logger.info(f"✓ Gate 3 VERIFY triggered (Weight delta: {reading.weight_delta}g)")
    
    def _success_gate(self, reading: SensorReading) -> None:
        """All gates passed - successful pick."""
        self.state = FSMState.SUCCESS
        logger.info(f"✓✓✓ PICK VERIFIED: {self.last_bin_id} - Updating inventory")
        # TODO: Call inventory update logic
    
    def _fail_gate(self, reading: SensorReading) -> None:
        """Gate timeout or condition failure."""
        self.state = FSMState.ERROR
        self.gate_entry_time = time.time()
        logger.warning(f"✗ Gate failed at state {self.previous_state}")
        # TODO: Trigger visual alert to operator
    
    def _reset_to_idle(self) -> None:
        """Reset FSM to idle state."""
        self.state = FSMState.IDLE
        self.gate_entry_time = None
        self.last_bin_id = None
    
    def _check_gate_1_timeout(self, current_time: float) -> bool:
        """Check if Gate 1 timeout exceeded."""
        return (current_time - self.gate_entry_time) > self.config['fsm']['gate1_spatial_timeout']
    
    def _check_gate_2_timeout(self, current_time: float) -> bool:
        """Check if Gate 2 timeout exceeded."""
        return (current_time - self.gate_entry_time) > self.config['fsm']['gate2_intent_timeout']
    
    def _check_gate_3_timeout(self, current_time: float) -> bool:
        """Check if Gate 3 timeout exceeded."""
        return (current_time - self.gate_entry_time) > self.config['fsm']['gate3_verification_timeout']
    
    def get_state_info(self) -> dict:
        """Get current FSM state information."""
        elapsed = time.time() - self.gate_entry_time if self.gate_entry_time else 0
        
        return {
            'state': self.state.value,
            'elapsed_time': elapsed,
            'bin_id': self.last_bin_id,
            'timestamp': time.time(),
        }
