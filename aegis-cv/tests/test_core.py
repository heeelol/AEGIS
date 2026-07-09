"""
Unit tests and mock sensor data for AEGIS Core.
"""

import pytest
from src.engine.fsm import TripleGateFSM, FSMState, SensorReading
from src.detectors.geofencing import GeofenceManager


@pytest.fixture
def fsm_config():
    """Fixture for FSM configuration."""
    return {
        'fsm': {
            'gate1_spatial_timeout': 2.0,
            'gate2_intent_timeout': 1.5,
            'gate3_verification_timeout': 3.0,
            'max_retries': 3,
            'error_cooldown': 5.0,
            'weight_delta_threshold': 50,
        }
    }


@pytest.fixture
def bins_config():
    """Fixture for bin geofence configuration."""
    return {
        'geofence_margin': 5,
        'bins': {
            'bin_0_0': {
                'x_min': 10, 'x_max': 120,
                'y_min': 10, 'y_max': 120,
            },
            'bin_0_1': {
                'x_min': 130, 'x_max': 240,
                'y_min': 10, 'y_max': 120,
            },
        }
    }


def test_fsm_initial_state(fsm_config):
    """Test FSM starts in IDLE state."""
    fsm = TripleGateFSM(fsm_config)
    assert fsm.state == FSMState.IDLE


def test_fsm_gate1_entry(fsm_config):
    """Test FSM enters Gate 1 when hand in geofence."""
    fsm = TripleGateFSM(fsm_config)
    
    reading = SensorReading(
        timestamp=0.0,
        hand_in_geofence=True,
        closed_fist_detected=False,
        weight_delta=0.0,
        bin_id='bin_0_0'
    )
    
    fsm.update(reading)
    assert fsm.state == FSMState.GATE_1_SPATIAL


def test_geofence_point_detection(bins_config):
    """Test geofence boundary checking."""
    manager = GeofenceManager(bins_config)
    
    # Point inside bin_0_0
    is_in, bin_id = manager.check_point_in_geofence(65, 65)
    assert is_in and bin_id == 'bin_0_0'
    
    # Point outside all bins
    is_in, bin_id = manager.check_point_in_geofence(1000, 1000)
    assert not is_in


if __name__ == '__main__':
    pytest.main([__file__])
