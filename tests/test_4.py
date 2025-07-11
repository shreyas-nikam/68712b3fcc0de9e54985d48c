import pytest
from definition_bc15bd687ec14cea8aa866ba6f045cbe import combine_distributions_quantile_avg_constant_weights

import numpy as np
from scipy.stats import norm


def ild_quantile_func(q):
    return norm.ppf(q, loc=100, scale=20)

def scenario_quantile_func(q):
    return norm.ppf(q, loc=150, scale=30)

@pytest.mark.parametrize("n_ild, m_scenario, quantiles_to_evaluate, expected", [
    (100, 50, [0.5], [np.power(ild_quantile_func(0.5), 2/3) * np.power(scenario_quantile_func(0.5), 1/3)]),
    (50, 100, [0.25, 0.75], [np.power(ild_quantile_func(0.25), 1/3) * np.power(scenario_quantile_func(0.25), 2/3),
                         np.power(ild_quantile_func(0.75), 1/3) * np.power(scenario_quantile_func(0.75), 2/3)]),
    (0, 100, [0.99], [scenario_quantile_func(0.99)]),
    (100, 0, [0.01], [ild_quantile_func(0.01)]),
    (100, 50, [], [])
])
def test_combine_distributions_quantile_avg_constant_weights(n_ild, m_scenario, quantiles_to_evaluate, expected):
    result = combine_distributions_quantile_avg_constant_weights(ild_quantile_func, scenario_quantile_func, n_ild, m_scenario, quantiles_to_evaluate)
    if len(expected) == 0:
        assert result == None
    else:
        assert np.allclose(result, expected, rtol=1e-5)
