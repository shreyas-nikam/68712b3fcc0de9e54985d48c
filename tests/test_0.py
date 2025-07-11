import pytest
from definition_c72cf6b75e874731a3a4f706efd08ab3 import fit_scenario_distribution
import numpy as np

def is_valid_distribution(dist):
    # Simple check if dist is a valid scipy.stats distribution.
    return hasattr(dist, 'rvs') and hasattr(dist, 'cdf')

def is_valid_distribution_output(output):
    dist, params = output
    return is_valid_distribution(dist) and isinstance(params, tuple)

@pytest.mark.parametrize("frequency, percentile_50, percentile_90, percentile_99", [
    (1, 100, 1000, 10000),
    (0.5, 50, 500, 5000),
    (2, 200, 2000, 20000),
    (1, 100, 100, 100),
    (1, 100, 200, np.inf)
])
def test_fit_scenario_distribution_valid_input(frequency, percentile_50, percentile_90, percentile_99):
    result = fit_scenario_distribution(frequency, percentile_50, percentile_90, percentile_99)
    assert is_valid_distribution_output(result)