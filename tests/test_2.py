import pytest
from definition_6859d4a559704ff1b15890d7ead57250 import fit_ild_distribution
import pandas as pd
import numpy as np
from scipy import stats

@pytest.fixture
def sample_ild_data():
    # Create a simple synthetic ILD dataset for testing
    return pd.Series(np.random.lognormal(mean=2, sigma=1, size=100))

def test_fit_ild_distribution_parametric(sample_ild_data):
    # Test fitting a Log-Normal distribution
    distribution, params = fit_ild_distribution(sample_ild_data, "lognorm", None)
    assert isinstance(distribution, type(stats.lognorm)), "Expected a lognorm distribution object"
    assert isinstance(params, tuple), "Expected parameters to be a tuple"
    # Cannot directly check for value equivalency due to randomness of fit, but check existence

def test_fit_ild_distribution_body_tail(sample_ild_data):
    # Test fitting a body-tail distribution (empirical body, GPD tail)
    threshold = sample_ild_data.quantile(0.9)
    distribution, params = fit_ild_distribution(sample_ild_data, "bodytail", threshold)
    assert isinstance(distribution, tuple), "Expected body and tail distribution object in a tuple"
    assert isinstance(params, tuple), "Expected parameters for body and tail distribution in a tuple"

def test_fit_ild_distribution_empty_data():
    # Test with empty data
    empty_data = pd.Series([])
    with pytest.raises(ValueError):
        fit_ild_distribution(empty_data, "lognorm", None)

def test_fit_ild_distribution_invalid_distribution_type(sample_ild_data):
     # Test with invalid distribution type
    with pytest.raises(ValueError, match="Invalid distribution type"):
        fit_ild_distribution(sample_ild_data, "invalid_dist", None)

def test_fit_ild_distribution_threshold_too_high(sample_ild_data):
    #Test with threshold that exceeds max data
    threshold = sample_ild_data.max() + 1
    with pytest.raises(ValueError):
        fit_ild_distribution(sample_ild_data, "bodytail", threshold)
