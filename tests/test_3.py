import pytest
from definition_a7bbef4ed4a740a9949e3ff33d1150b4 import combine_distributions_param_avg

@pytest.mark.parametrize("ild_dist_params, scenario_dist_params, n_ild, m_scenario, expected", [
    (0.5, 0.7, 100, 50, 0.5666666666666667),
    (0.0, 1.0, 50, 50, 0.5),
    (0.8, 0.8, 20, 30, 0.8),
    (0.2, 0.9, 10, 90, 0.83),
    (0.7, 0.3, 75, 25, 0.6)
])
def test_combine_distributions_param_avg(ild_dist_params, scenario_dist_params, n_ild, m_scenario, expected):
    assert combine_distributions_param_avg(ild_dist_params, scenario_dist_params, n_ild, m_scenario) == expected
