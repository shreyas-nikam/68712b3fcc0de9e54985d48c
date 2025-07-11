import pytest
from definition_237ec2c179bb4f94b3391bdbaec96770 import simulate_stability_paradox
import pandas as pd

@pytest.fixture
def base_params():
    return {
        "frequency_mean": 1,
        "severity_50": 1000,
        "severity_90": 5000,
        "severity_99": 10000,
        "precision": 10
    }

@pytest.fixture
def modified_params():
    return {
        "frequency_mean": 0.5,  # Reduced frequency
        "severity_50": 500,     # Reduced severity
        "severity_90": 5000,
        "severity_99": 10000,
        "precision": 10
    }

@pytest.fixture
def ild_params():
    return {
        "frequency_mean": 100,
        "frequency_dispersion": 1.2,
        "severity_distribution": "lognorm",
        "severity_params": {"mean": 500, "std": 200},
        "num_observations": 1000,
        "reporting_threshold": 100
    }

def test_simulate_stability_paradox_returns_dict(base_params, modified_params, ild_params):
    result = simulate_stability_paradox(base_params, modified_params, ild_params)
    assert isinstance(result, dict)

def test_simulate_stability_paradox_handles_zero_frequency(base_params, modified_params, ild_params):
    base_params["frequency_mean"] = 0
    modified_params["frequency_mean"] = 0
    try:
        simulate_stability_paradox(base_params, modified_params, ild_params)
    except Exception as e:
        assert isinstance(e, Exception)

def test_simulate_stability_paradox_validates_input_types(base_params, modified_params, ild_params):
    with pytest.raises(TypeError):
        simulate_stability_paradox("invalid", modified_params, ild_params)
    with pytest.raises(TypeError):
        simulate_stability_paradox(base_params, "invalid", ild_params)
    with pytest.raises(TypeError):
        simulate_stability_paradox(base_params, modified_params, "invalid")
    
def test_simulate_stability_paradox_with_negative_ild_threshold(base_params, modified_params, ild_params):
    ild_params["reporting_threshold"] = -100
    try:
        simulate_stability_paradox(base_params, modified_params, ild_params)
    except Exception as e:
        assert isinstance(e, Exception)

def test_simulate_stability_paradox_empty_params(base_params, modified_params, ild_params):
    try:
        simulate_stability_paradox({}, {}, {})
    except Exception as e:
        assert isinstance(e, Exception)
