import scipy.stats as stats
import numpy as np

def fit_scenario_distribution(frequency, percentile_50, percentile_90, percentile_99):
    """Fits a distribution to scenario data.

    Args:
        frequency: Expected annual frequency.
        percentile_50: 50th percentile loss.
        percentile_90: 90th percentile loss.
        percentile_99: 99th percentile loss.

    Returns:
        Fitted distribution object and its parameters.
    """
    # Using lognorm as an example distribution
    # Convert percentiles to log scale
    percentile_50_log = np.log(percentile_50)
    percentile_90_log = np.log(percentile_90)
    percentile_99_log = np.log(percentile_99)
    
    # Estimate parameters (example method, can be improved)
    s = 1  # Example value
    try:
        loc = percentile_50_log
        scale = np.exp(percentile_50_log) 
    except:
        loc = 0
        scale = 1

    return stats.lognorm(s, loc=0, scale=scale), (s, loc, scale)

import pandas as pd
import numpy as np
from scipy.stats import poisson, lognorm, pareto, genpareto

def generate_synthetic_ild(frequency_params, severity_params, num_observations, reporting_threshold):
    """Generates synthetic operational loss data."""
    if not isinstance(frequency_params, dict):
        raise TypeError("frequency_params must be a dictionary")
    if not isinstance(severity_params, dict):
        raise TypeError("severity_params must be a dictionary")
    
    losses = []
    for i in range(num_observations):
        # Generate number of losses for the year
        if frequency_params['distribution'] == 'poisson':
            num_losses = poisson.rvs(frequency_params['lambda'])
        else:
            raise ValueError("Invalid frequency distribution")

        # Generate severity amounts for each loss
        severity_amounts = []
        if severity_params['distribution'] == 'lognorm':
            shape = severity_params['sigma']
            loc = 0
            scale = np.exp(severity_params['mean'])
            severity_amounts = lognorm.rvs(s=shape, loc=loc, scale=scale, size=num_losses)
        elif severity_params['distribution'] == 'pareto':
             b = severity_params['shape']
             severity_amounts = pareto.rvs(b, loc=severity_params['loc'], scale=severity_params['scale'], size=num_losses)
        elif severity_params['distribution'] == 'gpd':
             c = severity_params['shape']
             severity_amounts = genpareto.rvs(c, loc=severity_params['loc'], scale=severity_params['scale'], size=num_losses)
        else:
            raise ValueError("Invalid severity distribution")

        losses.extend(severity_amounts)

    # Filter losses based on reporting threshold
    losses = [loss for loss in losses if loss >= reporting_threshold]

    # Create DataFrame
    df = pd.DataFrame({'Amount': losses})
    df['Loss_ID'] = range(1, len(df) + 1)
    df = df[['Loss_ID', 'Amount']]

    return df

import pandas as pd
import numpy as np
from scipy import stats

def fit_ild_distribution(ild_data, distribution_type, threshold):
    """Fits a distribution to ILD data."""

    if ild_data.empty:
        raise ValueError("ILD data cannot be empty.")

    if distribution_type == "lognorm":
        # Fit Log-Normal distribution
        shape, loc, scale = stats.lognorm.fit(ild_data, loc=0)
        distribution = stats.lognorm
        params = (shape, loc, scale)
        return distribution, params
    elif distribution_type == "bodytail":
        # Fit body-tail distribution (empirical body, GPD tail)
        if threshold is None:
            raise ValueError("Threshold must be specified for body-tail distribution.")
        if threshold > ild_data.max():
            raise ValueError("Threshold exceeds maximum value in data.")

        #GPD only for tail
        tail_data = ild_data[ild_data > threshold]
        if len(tail_data) == 0:
             raise ValueError("Threshold too high: No data in the tail.")

        # Fit GPD to the tail
        params = stats.genpareto.fit(tail_data, loc=threshold) #fitting GPD to tail data

        distribution = (stats.rv_histogram(np.histogram(ild_data[ild_data <= threshold], bins='auto')), stats.genpareto) #returning distributions

        return distribution, params
    else:
        raise ValueError("Invalid distribution type. Choose 'lognorm' or 'bodytail'.")

def combine_distributions_param_avg(ild_dist_params, scenario_dist_params, n_ild, m_scenario):
                """If both ILD and scenario severities are approximated by Pareto-like tails, this function calculates a combined tail parameter using the weighted average formula."""
                return (ild_dist_params * n_ild + scenario_dist_params * m_scenario) / (n_ild + m_scenario)

import numpy as np

def combine_distributions_quantile_avg_constant_weights(ild_quantile_func, scenario_quantile_func, n_ild, m_scenario, quantiles_to_evaluate):
    """Combines quantile functions using geometric average with constant weights.
    """
    if not quantiles_to_evaluate:
        return None

    total_precision = n_ild + m_scenario
    if total_precision == 0:
        return [np.nan] * len(quantiles_to_evaluate)  # Or handle appropriately

    combined_quantiles = []
    for q in quantiles_to_evaluate:
        if n_ild == 0:
            combined_quantiles.append(scenario_quantile_func(q))
        elif m_scenario == 0:
            combined_quantiles.append(ild_quantile_func(q))
        else:
            weight_ild = n_ild / total_precision
            weight_scenario = m_scenario / total_precision
            combined_quantile = np.power(ild_quantile_func(q), weight_ild) * np.power(scenario_quantile_func(q), weight_scenario)
            combined_quantiles.append(combined_quantile)

    return combined_quantiles

import numpy as np

def combine_distributions_quantile_avg_variable_weights(ild_quantile_func, scenario_quantile_func, n_ild, m_scenario, quantiles_to_evaluate, num_bootstraps):
    """Combines distributions using quantile averaging with variable weights."""
    combined_quantiles = []
    for q in quantiles_to_evaluate:
        # Bootstrapping to estimate variance
        ild_estimates = [ild_quantile_func(q) for _ in range(num_bootstraps)]
        scenario_estimates = [scenario_quantile_func(q) for _ in range(num_bootstraps)]

        # Variance calculation
        ild_var = np.var(ild_estimates) if num_bootstraps > 1 else 1.0
        scenario_var = np.var(scenario_estimates) if num_bootstraps > 1 else 1.0

        # Weight calculation based on inverse variance
        ild_weight = 1.0 / (ild_var + 1e-9)  # Adding a small constant to avoid division by zero
        scenario_weight = 1.0 / (scenario_var + 1e-9)

        # Normalize weights
        total_weight = ild_weight + scenario_weight
        ild_weight /= total_weight
        scenario_weight /= total_weight

        # Weighted geometric average of quantiles
        ild_quantile = ild_quantile_func(q)
        scenario_quantile = scenario_quantile_func(q)
        
        if ild_quantile <= 0 or scenario_quantile <= 0:
            combined_quantile = (ild_weight * ild_quantile) + (scenario_weight * scenario_quantile)

        else:
            combined_quantile = (ild_quantile**ild_weight) * (scenario_quantile**scenario_weight)

        combined_quantiles.append(combined_quantile)
    return combined_quantiles

def calculate_capital(combined_distribution_or_quantiles, confidence_level):
                """Estimates risk capital (e.g., 99.9% VaR, Expected Shortfall)."""
                if not isinstance(confidence_level, (int, float)):
                    return None
                if confidence_level <= 0 or confidence_level >= 1:
                    return None
                if not combined_distribution_or_quantiles:
                    return None
                if isinstance(combined_distribution_or_quantiles, (int, float)):
                    return combined_distribution_or_quantiles * confidence_level
                if isinstance(combined_distribution_or_quantiles, list):
                    if not combined_distribution_or_quantiles:
                        return None
                    
                    sorted_quantiles = sorted(combined_distribution_or_quantiles)
                    index = int(confidence_level * (len(sorted_quantiles) - 1))
                    return sorted_quantiles[index]
                return None

import pandas as pd
import numpy as np
from scipy.stats import lognorm, norm

def simulate_stability_paradox(base_scenario_params, modified_body_scenario_params, ild_params):
    """Simulates stability paradox and returns results."""

    # Input validation
    if not all(isinstance(param, dict) for param in [base_scenario_params, modified_body_scenario_params, ild_params]):
        raise TypeError("All parameters must be dictionaries.")

    def simulate_scenario(params):
        """Simulates a single scenario."""
        frequency_mean = params.get("frequency_mean")
        severity_50 = params.get("severity_50")
        severity_90 = params.get("severity_90")
        severity_99 = params.get("severity_99")
        precision = params.get("precision", 10)
        
        if frequency_mean is None or severity_50 is None or severity_90 is None or severity_99 is None:
            raise ValueError("Missing parameters in scenario definition.")

        if frequency_mean < 0:
            raise ValueError("Frequency cannot be negative.")

        num_simulations = 10000
        losses = []

        if frequency_mean > 0:  # only simulate losses if frequency is positive
            for _ in range(num_simulations):
                num_losses = np.random.poisson(frequency_mean)
                for _ in range(num_losses):
                    # Simple severity simulation (can be improved with more sophisticated distributions)
                    u = np.random.uniform()
                    if u < 0.5:
                        loss = np.random.uniform(0, severity_50)
                    elif u < 0.9:
                        loss = np.random.uniform(severity_50, severity_90)
                    else:
                        loss = np.random.uniform(severity_90, severity_99)

                    losses.append(loss)
        return losses

    def simulate_ild(params):
        """Simulates ILD data."""
        frequency_mean = params.get("frequency_mean")
        frequency_dispersion = params.get("frequency_dispersion")
        severity_distribution = params.get("severity_distribution")
        severity_params = params.get("severity_params")
        num_observations = params.get("num_observations")
        reporting_threshold = params.get("reporting_threshold")
        
        if frequency_mean is None or frequency_dispersion is None or severity_distribution is None or severity_params is None or num_observations is None or reporting_threshold is None:
            raise ValueError("Missing parameters in ILD definition.")

        if reporting_threshold < 0:
            raise ValueError("Reporting threshold should not be negative")
        
        ild_losses = []
        for _ in range(num_observations):
            num_losses = np.random.poisson(frequency_mean)
            for _ in range(num_losses):
                if severity_distribution == "lognorm":
                    s = np.random.lognormal(mean=np.log(severity_params["mean"]), sigma=severity_params["std"]/severity_params["mean"])
                else: # Default to normal
                    s = np.random.normal(loc=severity_params["mean"], scale=severity_params["std"])
                if s > reporting_threshold:
                    ild_losses.append(s)
        return ild_losses
    
    # Simulate base and modified scenarios
    base_losses = simulate_scenario(base_scenario_params)
    modified_losses = simulate_scenario(modified_body_scenario_params)

    # Simulate ILD
    ild_losses = simulate_ild(ild_params)

    # Combine losses
    combined_base_losses = base_losses + ild_losses
    combined_modified_losses = modified_losses + ild_losses
    
    # Calculate capital (e.g., VaR 99.5%)
    def calculate_var(losses, confidence_level=0.995):
          if not losses:
              return 0  # or some other appropriate value, like the reporting threshold for ILD
          return np.quantile(losses, confidence_level)

    var_base = calculate_var(combined_base_losses)
    var_modified = calculate_var(combined_modified_losses)

    return {
        "base_var": var_base,
        "modified_var": var_modified,
        "base_losses": combined_base_losses,
        "modified_losses": combined_modified_losses
    }