
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import lognorm, norm, genpareto
from plotly.subplots import make_subplots

def combine_distributions_param_avg(ild_dist_params, scenario_dist_params, n_ild, m_scenario):
    """If both ILD and scenario severities are approximated by Pareto-like tails, this function calculates a combined tail parameter using the weighted average formula."""
    combined_tail_param = (ild_dist_params * n_ild + scenario_dist_params * m_scenario) / (n_ild + m_scenario)
    return combined_tail_param

def combine_distributions_quantile_avg_constant_weights(ild_quantile_func, scenario_quantile_func, n_ild, m_scenario, quantiles_to_evaluate):
    """Combines quantile functions using geometric average with constant weights."""
    if not quantiles_to_evaluate:
        return None

    total_precision = n_ild + m_scenario
    if total_precision == 0:
        return [np.nan] * len(quantiles_to_evaluate)

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
            # Fallback to arithmetic mean if quantiles are non-positive, as geometric mean requires positive values
            combined_quantile = (ild_weight * ild_quantile) + (scenario_weight * scenario_quantile)
        else:
            combined_quantile = (ild_quantile**ild_weight) * (scenario_quantile**scenario_weight)

        combined_quantiles.append(combined_quantile)
    return combined_quantiles

def calculate_capital(combined_distribution_or_quantiles, confidence_level):
    """Estimates risk capital (e.g., 99.9\% VaR, Expected Shortfall)."""
    if not isinstance(confidence_level, (int, float)):
        return None
    if confidence_level <= 0 or confidence_level >= 1:
        return None
    if not combined_distribution_or_quantiles:
        return None
    if isinstance(combined_distribution_or_quantiles, (int, float)):
        # This simplified scaling might need re-evaluation for a direct VaR input
        return combined_distribution_or_quantiles * confidence_level
    if isinstance(combined_distribution_or_quantiles, list):
        if not combined_distribution_or_quantiles:
            return None

        sorted_quantiles = sorted(combined_distribution_or_quantiles)
        index = int(confidence_level * (len(sorted_quantiles) - 1))
        return sorted_quantiles[index]
    return None

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

        num_simulations = 10000 # Hardcoded for simulation, consider making it a UI input
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
        frequency_dispersion = params.get("frequency_dispersion") # Not used in current notebook code for ILD simulation, but could be for other dist.
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

    # Calculate capital (e.g., VaR 99.5\%)
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


def run_page3():
    st.header("Distribution Combination and Stability Paradox Simulation")
    st.markdown("""
    This section explores methods for combining different risk distributions (scenario and ILD) and demonstrates the 'stability paradox'.
    The stability paradox highlights how seemingly beneficial changes in the body of a risk distribution can paradoxically increase overall risk.
    """)

    st.subheader("1. Combining Distributions - Parameter Averaging")
    st.markdown("""
    This method combines distributions by averaging their parameters, particularly relevant when both ILD and scenario severities are approximated by Pareto-like tails.
    It uses a weighted average based on the precision (number of losses) of each source.
    """)

    col1, col2 = st.columns(2)
    with col1:
        ild_tail_param = st.number_input("ILD Tail Parameter (ξ1)", value=0.5, min_value=0.0, help="Shape parameter for ILD's Pareto-like tail.")
        n_ild = st.number_input("ILD Precision (n)", value=100, min_value=1, help="Effective number of losses representing ILD precision.")
    with col2:
        scenario_tail_param = st.number_input("Scenario Tail Parameter (ξ2)", value=0.7, min_value=0.0, help="Shape parameter for Scenario's Pareto-like tail.")
        m_scenario = st.number_input("Scenario Precision (m)", value=10, min_value=1, help="Effective number of losses representing scenario precision.")

    combined_tail_param = combine_distributions_param_avg(ild_tail_param, scenario_tail_param, n_ild, m_scenario)
    st.write(f"**Combined Tail Parameter (ξ1,2):** {combined_tail_param:.4f}")
    st.markdown("""
    The combined tail parameter is calculated as a weighted average of the individual tail parameters. The weights are determined by the precision of each data source.
    A higher precision for one data source will give its tail parameter more influence on the combined parameter.
    """)

    st.subheader("2. Combining Distributions - Quantile Averaging with Constant Weights")
    st.markdown("""
    This method combines the quantiles of individual ILD and scenario distributions using a geometric average with constant weights.
    """)
    quantiles_to_evaluate = st.multiselect("Quantiles to Evaluate (Constant Weights)", [0.90, 0.95, 0.99, 0.995], default=[0.90, 0.95, 0.99], help="Select the percentiles for which to calculate combined quantiles.")

    # Dummy quantile functions for demonstration
    def ild_quantile_func(q): return lognorm.ppf(q, s=0.7, scale=1000) # Dummy Log-Normal ILD
    def scenario_quantile_func(q): return lognorm.ppf(q, s=0.8, scale=5000) # Dummy Log-Normal Scenario

    combined_quantiles_constant = combine_distributions_quantile_avg_constant_weights(ild_quantile_func, scenario_quantile_func, n_ild, m_scenario, quantiles_to_evaluate)

    if combined_quantiles_constant:
        df_quantiles = pd.DataFrame({
            "Quantile": quantiles_to_evaluate,
            "ILD Quantile": [ild_quantile_func(q) for q in quantiles_to_evaluate],
            "Scenario Quantile": [scenario_quantile_func(q) for q in quantiles_to_evaluate],
            "Combined Quantile": combined_quantiles_constant
        })
        st.dataframe(df_quantiles)

        # Plot Quantiles
        fig_quantiles = go.Figure()
        fig_quantiles.add_trace(go.Scatter(x=df_quantiles["Quantile"], y=df_quantiles["ILD Quantile"], mode='lines+markers', name='ILD Quantile'))
        fig_quantiles.add_trace(go.Scatter(x=df_quantiles["Quantile"], y=df_quantiles["Scenario Quantile"], mode='lines+markers', name='Scenario Quantile'))
        fig_quantiles.add_trace(go.Scatter(x=df_quantiles["Quantile"], y=df_quantiles["Combined Quantile"], mode='lines+markers', name='Combined Quantile'))
        fig_quantiles.update_layout(title="Quantile Averaging with Constant Weights", xaxis_title="Quantile", yaxis_title="Loss Amount", font=dict(size=12))
        st.plotly_chart(fig_quantiles, use_container_width=True)

    st.subheader("3. Simulating the Stability Paradox")
    st.markdown("""
    This section demonstrates the 'stability paradox' where seemingly positive changes in the 'body' of a risk distribution can paradoxically lead to an increase in overall tail risk and implied capital.
    """)

    with st.expander("Base Scenario Parameters"):
        base_scenario_params = {
            "frequency_mean": st.number_input("Base Frequency", value=1.0, help="Expected number of base scenario losses per year."),
            "severity_50": st.number_input("Base 50th Percentile", value=1_000, help="Base 50th percentile loss amount."),
            "severity_90": st.number_input("Base 90th Percentile", value=5_000, help="Base 90th percentile loss amount."),
            "severity_99": st.number_input("Base 99th Percentile", value=10_000, help="Base 99th percentile loss amount.")
        }
    with st.expander("Modified Body Scenario Parameters"):
        modified_body_scenario_params = {
            "frequency_mean": st.number_input("Modified Frequency", value=0.5, help="Expected number of modified scenario losses per year."),
            "severity_50": st.number_input("Modified 50th Percentile", value=1_000, help="Modified 50th percentile loss amount."),
            "severity_90": st.number_input("Modified 90th Percentile", value=5_000, help="Modified 90th percentile loss amount."),
            "severity_99": st.number_input("Modified 99th Percentile", value=10_000, help="Modified 99th percentile loss amount.")
        }
    with st.expander("ILD Parameters for Paradox"):
        ild_params_paradox = {
            "frequency_mean": st.number_input("ILD Frequency", value=10.0, help="Expected number of ILD losses per year."),
            "frequency_dispersion": 1.0, # Not used in the notebook
            "severity_distribution": "lognorm",
            "severity_params": {"mean": 500, "std": 200},
            "num_observations": st.number_input("ILD Observations", value=10, help="Number of years of ILD data."),
            "reporting_threshold": st.number_input("ILD Reporting Threshold", value=100, help="Minimum ILD loss to record.")
        }

    stability_results = simulate_stability_paradox(base_scenario_params, modified_body_scenario_params, ild_params_paradox)

    if stability_results:
        st.write(f"**Base Scenario VaR (99.5\%):** {stability_results['base_var']:.2f}")
        st.write(f"**Modified Scenario VaR (99.5\%):** {stability_results['modified_var']:.2f}")

        if stability_results['modified_var'] > stability_results['base_var']:
            st.warning("Stability Paradox Observed: Modified scenario has a higher VaR than the base scenario, even though the body of the distribution improved.")
        else:
            st.success("No Stability Paradox Observed: Modified scenario has a lower VaR than the base scenario.")

        # Plotting
        fig_paradox = go.Figure()

        # Plotting histograms of losses
        fig_paradox.add_trace(go.Histogram(x=stability_results['base_losses'], name='Base Scenario Losses', opacity=0.6))
        fig_paradox.add_trace(go.Histogram(x=stability_results['modified_losses'], name='Modified Scenario Losses', opacity=0.6))

        fig_paradox.update_layout(barmode='overlay', title="Stability Paradox Simulation", xaxis_title="Loss Amount", yaxis_title="Frequency", font=dict(size=12))
        st.plotly_chart(fig_paradox, use_container_width=True)
