
import streamlit as st
import scipy.stats as stats
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    # This is a simplified fitting process for demonstration.
    # A more robust fitting would involve more data points or optimization.

    # Approximating parameters based on percentiles for lognorm
    # log(median) = mu
    # log(90th percentile) = mu + sigma * inv_cdf(0.9)
    # log(99th percentile) = mu + sigma * inv_cdf(0.99)
    # Using 50th and 90th percentile to estimate mu and sigma roughly

    try:
        # Using a two-point estimation for lognorm parameters (simplified)
        # For lognormal, P(X <= x) = P(ln(X) <= ln(x))
        # So, ln(x) is normally distributed with mean=mu_log, std=sigma_log
        # We have ln(x_p) = mu_log + sigma_log * norm.ppf(p)

        # From 50th percentile: ln(percentile_50) = mu_log + sigma_log * norm.ppf(0.5) = mu_log
        mu_log = np.log(percentile_50)

        # From 90th percentile: ln(percentile_90) = mu_log + sigma_log * norm.ppf(0.9)
        # sigma_log = (ln(percentile_90) - mu_log) / norm.ppf(0.9)
        # Ensure percentile_90 > percentile_50
        if percentile_90 <= percentile_50:
            st.error("90th Percentile must be greater than 50th Percentile.")
            return None, None
        sigma_log = (np.log(percentile_90) - mu_log) / stats.norm.ppf(0.9)

        if sigma_log <= 0:
            st.error("Calculated sigma_log is not positive. Please check percentile values.")
            return None, None

        # The parameters for scipy.stats.lognorm are s (shape), loc (location), scale
        # s = sigma_log
        # scale = exp(mu_log) (this is the geometric mean)
        # loc = 0 (standard assumption for loss distributions)

        s_param = sigma_log
        loc_param = 0 # Assuming losses are positive
        scale_param = np.exp(mu_log)

        # Validate with 99th percentile (optional, for consistency check)
        # expected_99th_log = mu_log + sigma_log * stats.norm.ppf(0.99)
        # expected_99th = np.exp(expected_99th_log)
        # st.write(f"Consistency check: Expected 99th percentile based on 50th/90th fit: {expected_99th:,.2f}")

        return stats.lognorm(s=s_param, loc=loc_param, scale=scale_param), (s_param, loc_param, scale_param)
    except Exception as e:
        st.error(f"Error fitting distribution: {e}. Please ensure percentiles are valid and increasing.")
        return None, None


def run_page1():
    st.header("Scenario Distribution Fitting")
    st.markdown("""
    This section allows you to define hypothetical operational risk scenarios by specifying
    their expected frequency and various loss percentiles. The application will then fit a
    statistical distribution (Log-Normal in this case) to these expert assessments,
    providing a quantitative representation of the scenario's potential impact.

    This process is crucial for converting qualitative risk insights into a quantifiable form
    that can be used in aggregated risk models.
    """)

    st.subheader("Define Scenario Parameters")
    col1, col2 = st.columns(2)
    with col1:
        frequency = st.number_input(
            "Expected Annual Frequency (Scenario)",
            value=1.0,
            min_value=0.01,
            help="Expected number of loss events per year for the scenario."
        )
    with col2:
        percentile_50 = st.number_input(
            "50th Percentile Loss (Scenario)",
            value=1_000_000.0,
            min_value=1.0,
            help="The loss value below which 50\% of scenario losses fall."
        )
        percentile_90 = st.number_input(
            "90th Percentile Loss (Scenario)",
            value=5_000_000.0,
            min_value=1.0,
            help="The loss value below which 90\% of scenario losses fall. Must be greater than 50th percentile."
        )
        percentile_99 = st.number_input(
            "99th Percentile Loss (Scenario)",
            value=10_000_000.0,
            min_value=1.0,
            help="The loss value below which 99\% of scenario losses fall. Must be greater than 90th percentile."
        )

    if not (percentile_90 > percentile_50 and percentile_99 > percentile_90):
        st.warning("Please ensure percentiles are strictly increasing: 50th < 90th < 99th.")
    else:
        st.subheader("Fitted Scenario Distribution")
        fitted_dist, params = fit_scenario_distribution(frequency, percentile_50, percentile_90, percentile_99)

        if fitted_dist and params:
            st.write(f"**Fitted Distribution Type:** Log-Normal")
            st.write(f"**Estimated Parameters (s, loc, scale):** s={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:,.2f}")

            st.markdown("""
            The Log-Normal distribution is characterized by its shape parameter $s$ (sigma of the underlying normal distribution),
            location parameter $\text{loc}$ (typically 0 for loss distributions), and scale parameter $\text{scale}$
            (geometric mean, $e^{\text{mu}}$ of the underlying normal distribution).
            """)
            st.latex(r"f(x; s, \text{loc}, \text{scale}) = \frac{1}{x s \sqrt{2\pi}} \exp\left(-\frac{(\ln(x - \text{loc}) - \ln(\text{scale}))^2}{2s^2}\right)")
            st.latex(r"\text{where } x > \text{loc}")
            st.latex(r"\text{s = sigma, scale = } e^{\text{mu}}")

            # Plot PDF and CDF
            x = np.linspace(max(0.1, fitted_dist.ppf(0.001)), fitted_dist.ppf(0.999), 500)
            pdf = fitted_dist.pdf(x)
            cdf = fitted_dist.cdf(x)

            fig = make_subplots(rows=1, cols=2, subplot_titles=('Probability Density Function (PDF)', 'Cumulative Distribution Function (CDF)'))

            fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color='blue')), row=1, col=1)
            fig.update_xaxes(title_text="Loss Amount", row=1, col=1)
            fig.update_yaxes(title_text="Density", row=1, col=1)

            fig.add_trace(go.Scatter(x=x, y=cdf, mode='lines', name='CDF', line=dict(color='red')), row=1, col=2)
            fig.update_xaxes(title_text="Loss Amount", row=1, col=2)
            fig.update_yaxes(title_text="Probability", row=1, col=2)

            fig.update_layout(height=400, showlegend=False, title_text="Fitted Scenario Distribution PDF and CDF",
                              font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretation of Plots:**
            *   **PDF (Probability Density Function):** Shows the relative likelihood of a loss occurring at a given amount. The peak indicates the most probable loss range.
            *   **CDF (Cumulative Distribution Function):** Shows the probability that a loss will be less than or equal to a given amount. For example, the point where the CDF reaches 0.90 corresponds to the 90th percentile loss.
            """)

        else:
            st.error("Could not fit scenario distribution with the provided parameters. Please adjust values.")

