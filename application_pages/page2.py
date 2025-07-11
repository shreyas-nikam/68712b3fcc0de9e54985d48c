
import streamlit as st
import scipy.stats as stats
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson, lognorm, pareto, genpareto
from plotly.subplots import make_subplots

@st.cache_data
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
            scale = np.exp(severity_params['mean']) # This assumes mean is geometric mean
            # If 'mean' is arithmetic mean, a more complex transformation is needed
            # For simplicity, let's assume 'mean' corresponds to the scale parameter (exp(mu))
            severity_amounts = lognorm.rvs(s=shape, loc=loc, scale=scale, size=num_losses)
        elif severity_params['distribution'] == 'pareto':
             b = severity_params['shape']
             # Ensure scale is positive
             scale_param = max(1e-9, severity_params['scale'])
             severity_amounts = pareto.rvs(b, loc=severity_params['loc'], scale=scale_param, size=num_losses)
        elif severity_params['distribution'] == 'gpd':
             c = severity_params['shape']
             # Ensure scale is positive
             scale_param = max(1e-9, severity_params['scale'])
             severity_amounts = genpareto.rvs(c, loc=severity_params['loc'], scale=scale_param, size=num_losses)
        else:
            raise ValueError("Invalid severity distribution")

        losses.extend(severity_amounts)

    # Filter losses based on reporting threshold
    losses = [loss for loss in losses if loss >= reporting_threshold]

    # Create DataFrame
    if losses:
        df = pd.DataFrame({'Amount': losses})
        df['Loss_ID'] = range(1, len(df) + 1)
        df = df[['Loss_ID', 'Amount']]
    else:
        df = pd.DataFrame(columns=['Loss_ID', 'Amount'])

    return df

@st.cache_data
def fit_ild_distribution(ild_data, distribution_type, threshold):
    """Fits a distribution to ILD data."""

    if ild_data.empty:
        st.warning("ILD data is empty. Cannot fit distribution.")
        return None, None

    if distribution_type == "lognorm":
        # Fit Log-Normal distribution
        # Ensure data is positive for lognorm fit
        positive_ild_data = ild_data[ild_data['Amount'] > 0]['Amount']
        if positive_ild_data.empty:
            st.warning("No positive loss amounts to fit Log-Normal distribution.")
            return None, None
        shape, loc, scale = stats.lognorm.fit(positive_ild_data, floc=0) # floc=0 fixes location at 0
        distribution = stats.lognorm
        params = (shape, loc, scale)
        return distribution, params
    elif distribution_type == "bodytail":
        # Fit body-tail distribution (empirical body, GPD tail)
        if threshold is None:
            raise ValueError("Threshold must be specified for body-tail distribution.")
        
        # Ensure threshold is within data range for a meaningful split
        if threshold >= ild_data['Amount'].max() and len(ild_data['Amount']) > 0:
            st.warning("Threshold is too high: All data points are below or equal to the threshold. Adjust threshold to be lower than the maximum loss amount for a meaningful tail.")
            return None, None
        if threshold <= ild_data['Amount'].min() and len(ild_data['Amount']) > 0:
            st.warning("Threshold is too low: All data points are above or equal to the threshold. Adjust threshold to be higher than the minimum loss amount for a meaningful body/tail split.")
            return None, None

        body_data = ild_data[ild_data['Amount'] <= threshold]['Amount']
        tail_data = ild_data[ild_data['Amount'] > threshold]['Amount']

        if len(tail_data) < 5: # Need a reasonable number of points for GPD fit
             st.warning(f"Not enough data points ({len(tail_data)}) in the tail for a robust GPD fit. Consider lowering the threshold or generating more ILD.")
             return None, None

        # Fit GPD to the tail
        # GPD parameters are c (shape), loc (location), scale
        try:
            # For GPD, location should typically be the threshold itself or 0 depending on definition
            # stats.genpareto.fit takes data, and optionally floc (fixed location) or initial guess
            gpd_params = stats.genpareto.fit(tail_data, floc=threshold)
            gpd_dist = stats.genpareto(*gpd_params)
        except Exception as e:
            st.error(f"Error fitting GPD to tail data: {e}. Check threshold and data distribution.")
            return None, None

        # For the body, we can represent it empirically with a histogram
        # Create bins for the body data
        if not body_data.empty:
            hist, bin_edges = np.histogram(body_data, bins='auto', density=True)
            body_dist = stats.rv_histogram((hist, bin_edges))
        else:
            body_dist = None # No body data to fit

        # Returning distributions and parameters for both body and tail
        # Note: the `distribution` here is a tuple: (body_distribution_object, tail_distribution_object)
        # params here is the gpd_params for the tail
        return (body_dist, gpd_dist), gpd_params
    else:
        raise ValueError("Invalid distribution type. Choose 'lognorm' or 'bodytail'.")


def run_page2():
    st.header("Synthetic ILD Generation and Fitting")
    st.markdown("""
    This section simulates historical operational losses (Internal Loss Data - ILD) and allows
    you to fit statistical distributions to this generated data. This is crucial for
    understanding the empirical characteristics of your loss experience and for
    extrapolating to potential future losses.
    """)

    st.subheader("1. Generate Synthetic ILD")
    st.markdown("""
    Configure the parameters below to generate a synthetic dataset of operational losses.
    You can specify the frequency and severity distribution characteristics, along with
    the number of simulation periods and a reporting threshold (minimum loss amount recorded).
    """)

    # ILD Generation Inputs
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Frequency Parameters**")
        frequency_dist_ild = st.selectbox(
            "ILD Frequency Distribution",
            ["poisson"],
            help="Choose the distribution for the number of losses per period."
        )
        frequency_lambda_ild = st.number_input(
            "ILD Frequency Rate (lambda)",
            value=100,
            min_value=1,
            help="Average number of loss events per period (for Poisson distribution)."
        )
        num_observations = st.slider(
            "Number of Simulation Periods (ILD)",
            value=5,
            min_value=1,
            max_value=20,
            help="Number of periods (e.g., years) to simulate ILD for."
        )
    with col2:
        st.markdown("**Severity Parameters**")
        severity_dist_ild = st.selectbox(
            "ILD Severity Distribution",
            ["lognorm", "pareto", "gpd"],
            help="Choose the distribution for individual loss amounts."
        )
        severity_params_ild = {}
        if severity_dist_ild == "lognorm":
            severity_params_ild["mean"] = st.number_input("Mean (lognorm)", value=10000.0, help="Geometric mean for log-normal severity.")
            severity_params_ild["sigma"] = st.number_input("Sigma (lognorm)", value=0.5, min_value=0.01, help="Standard deviation of log-transformed data for log-normal severity.")
        elif severity_dist_ild == "pareto":
            severity_params_ild["shape"] = st.number_input("Shape (Pareto)", value=1.0, min_value=0.01, help="Shape parameter 'b' for Pareto severity.")
            severity_params_ild["loc"] = st.number_input("Location (Pareto)", value=0.0, help="Location parameter for Pareto severity.")
            severity_params_ild["scale"] = st.number_input("Scale (Pareto)", value=1000.0, min_value=0.01, help="Scale parameter for Pareto severity.")
        elif severity_dist_ild == "gpd":
            severity_params_ild["shape"] = st.number_input("Shape (GPD)", value=0.1, help="Shape parameter 'c' for GPD severity.")
            severity_params_ild["loc"] = st.number_input("Location (GPD)", value=0.0, help="Location parameter for GPD severity.")
            severity_params_ild["scale"] = st.number_input("Scale (GPD)", value=1000.0, min_value=0.01, help="Scale parameter for GPD severity.")

        reporting_threshold = st.number_input(
            "ILD Reporting Threshold",
            value=1000.0,
            min_value=0.0,
            help="Only losses above this amount are recorded in ILD."
        )

    frequency_params_ild = {"distribution": frequency_dist_ild, "lambda": frequency_lambda_ild}
    severity_params_ild["distribution"] = severity_dist_ild # Add distribution type to severity params dict

    ild_data = generate_synthetic_ild(frequency_params_ild, severity_params_ild, num_observations, reporting_threshold)

    if not ild_data.empty:
        st.subheader("Generated ILD Data")
        st.write(f"Total {len(ild_data)} losses generated above the reporting threshold of {reporting_threshold:,.2f}.")
        st.dataframe(ild_data.head()) # Display first few rows
        st.write(f"**Summary Statistics:**")
        st.write(f"- Total Losses: ${ild_data['Amount'].sum():,.2f}")
        st.write(f"- Mean Loss Amount: ${ild_data['Amount'].mean():,.2f}")
        st.write(f"- Max Loss Amount: ${ild_data['Amount'].max():,.2f}")

        fig_hist = px.histogram(ild_data, x="Amount", nbins=50, title="Histogram of Generated ILD",
                                labels={'Amount': 'Loss Amount'}, height=400)
        fig_hist.update_layout(font=dict(size=12))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("No ILD data generated. Please check your frequency, severity parameters, and reporting threshold. "
                   "It's possible no losses were generated above the threshold.")

    st.subheader("2. Fit Distribution to ILD")
    st.markdown("""
    Now, fit a statistical distribution to the generated ILD. You can choose between a single
    Log-Normal distribution or a Body-Tail approach, where the body of the distribution
    is empirical and the tail is modeled by a Generalized Pareto Distribution (GPD).
    """)

    ild_fit_type = st.selectbox(
        "ILD Distribution Fit Type",
        ["lognorm", "bodytail"],
        help="Choose 'lognorm' for a single distribution fit, or 'bodytail' for a combined empirical/GPD fit."
    )
    threshold_bodytail = None
    if ild_fit_type == "bodytail":
        threshold_bodytail = st.number_input(
            "Threshold for Body-Tail Fit",
            value=ild_data['Amount'].quantile(0.75) if not ild_data.empty else 10000.0,
            min_value=0.0,
            help="Losses above this threshold are modeled by GPD (for 'bodytail' fit type)."
        )

    if not ild_data.empty:
        fitted_ild_dist, ild_fit_params = fit_ild_distribution(ild_data, ild_fit_type, threshold_bodytail)

        if fitted_ild_dist and ild_fit_params:
            st.write(f"**Fitted ILD Distribution Type:** {ild_fit_type.replace('lognorm', 'Log-Normal').replace('bodytail', 'Body-Tail (Empirical Body / GPD Tail)')}")

            if ild_fit_type == "lognorm":
                st.write(f"**Estimated Parameters (s, loc, scale):** s={ild_fit_params[0]:.4f}, loc={ild_fit_params[1]:.4f}, scale={ild_fit_params[2]:,.2f}")
                dist_to_plot = fitted_ild_dist
                st.markdown("""
                For Log-Normal fit, the parameters are $s$ (shape), $\text{loc}$ (location), and $\text{scale}$.
                """)

                x_min_plot = max(0.1, dist_to_plot.ppf(0.001))
                x_max_plot = dist_to_plot.ppf(0.999) if dist_to_plot.ppf(0.999) < ild_data['Amount'].max() * 2 else ild_data['Amount'].max() * 2 # avoid extremely large x-ranges
                x_plot = np.linspace(x_min_plot, x_max_plot, 500)
                pdf_plot = dist_to_plot.pdf(x_plot)
                cdf_plot = dist_to_plot.cdf(x_plot)

                fig_fit = make_subplots(rows=1, cols=2, subplot_titles=('Fitted PDF vs. ILD Histogram', 'Fitted CDF'))
                
                # PDF subplot
                fig_fit.add_trace(go.Histogram(x=ild_data['Amount'], histnorm='probability density', name='ILD Histogram', opacity=0.6), row=1, col=1)
                fig_fit.add_trace(go.Scatter(x=x_plot, y=pdf_plot, mode='lines', name='Fitted PDF', line=dict(color='blue', width=2)), row=1, col=1)
                fig_fit.update_xaxes(title_text="Loss Amount", row=1, col=1)
                fig_fit.update_yaxes(title_text="Density", row=1, col=1)
                
                # CDF subplot
                fig_fit.add_trace(go.Scatter(x=x_plot, y=cdf_plot, mode='lines', name='Fitted CDF', line=dict(color='red', width=2)), row=1, col=2)
                
                # Add empirical CDF of ILD data for comparison
                sorted_ild = np.sort(ild_data['Amount'])
                y_ecdf = np.arange(1, len(sorted_ild) + 1) / len(sorted_ild)
                fig_fit.add_trace(go.Scatter(x=sorted_ild, y=y_ecdf, mode='lines', name='ILD ECDF', line=dict(color='orange', dash='dash'), showlegend=True), row=1, col=2)

                fig_fit.update_xaxes(title_text="Loss Amount", row=1, col=2)
                fig_fit.update_yaxes(title_text="Probability", row=1, col=2)

                fig_fit.update_layout(height=450, showlegend=True, title_text="Fitted ILD Distribution (Log-Normal)",
                                      font=dict(size=12), legend=dict(x=0.01, y=0.99))
                st.plotly_chart(fig_fit, use_container_width=True)


            elif ild_fit_type == "bodytail":
                body_dist, gpd_dist = fitted_ild_dist
                gpd_params = ild_fit_params # shape, loc, scale for GPD
                st.write(f"**GPD Tail Parameters (c, loc, scale):** c={gpd_params[0]:.4f}, loc={gpd_params[1]:.4f}, scale={gpd_params[2]:,.2f}")
                st.markdown("""
                For the Body-Tail fit:
                *   The **body** of the distribution (losses $\leq$ threshold) is represented empirically (histogram).
                *   The **tail** of the distribution (losses $>$ threshold) is modeled by a Generalized Pareto Distribution (GPD).
                The GPD is characterized by its shape parameter $\xi$ ($c$), scale parameter $\sigma$ (scale), and location parameter $u$ (loc).
                """)
                st.latex(r"G_{\xi,\sigma}(x) = 1 - \left(1 + \frac{\xi (x-u)}{\sigma}\right)^{-1/\xi}")
                st.latex(r"\text{for } x > u")
                st.latex(r"\text{If } \xi = 0, G_{0,\sigma}(x) = 1 - e^{-(x-u)/\sigma}")


                # Plotting body and tail
                fig_bodytail = go.Figure()

                # Histogram of all ILD data for context
                fig_bodytail.add_trace(go.Histogram(x=ild_data['Amount'], histnorm='probability density', name='ILD Histogram (All)', opacity=0.5, marker_color='grey'))

                # Plot fitted GPD for tail
                if threshold_bodytail is not None and gpd_dist:
                    tail_data_max = ild_data['Amount'].max()
                    x_tail = np.linspace(threshold_bodytail, tail_data_max * 1.5, 200) # Extend beyond max data for visualization
                    # Filter for x > threshold to ensure GPD is plotted for excesses
                    x_tail = x_tail[x_tail > threshold_bodytail]
                    if len(x_tail) > 0:
                        pdf_gpd = gpd_dist.pdf(x_tail)
                        fig_bodytail.add_trace(go.Scatter(x=x_tail, y=pdf_gpd, mode='lines', name='Fitted GPD (Tail)', line=dict(color='purple', width=2)))
                        # Add a vertical line for the threshold
                        fig_bodytail.add_vline(x=threshold_bodytail, line_dash="dash", line_color="green", annotation_text=f"Threshold: {threshold_bodytail:,.0f}", annotation_position="top right")


                fig_bodytail.update_layout(title_text="Body-Tail ILD Distribution Fit",
                                           xaxis_title="Loss Amount", yaxis_title="Density",
                                           font=dict(size=12), height=450, showlegend=True,
                                           legend=dict(x=0.01, y=0.99))
                st.plotly_chart(fig_bodytail, use_container_width=True)

                st.markdown("""
                **Understanding the Body-Tail Fit Plot:**
                *   The grey histogram shows the empirical distribution of all generated ILD.
                *   The purple line represents the fitted GPD for losses exceeding the set threshold (green dashed line).
                    This visual helps confirm how well the GPD captures the extreme events in the tail.
                """)
        else:
            st.error("Could not fit ILD distribution with the provided parameters. Adjust threshold or generate more data.")
    else:
        st.info("Generate ILD data first to enable distribution fitting.")

# To make ild_data accessible across pages for combination functions,
# we can store it in st.session_state.
# This assumes run_page2 is called, and ild_data is generated.
# The code below is not part of run_page2 but a mechanism to pass data.
# This part would typically be handled by a global state management or passed as an argument.
# For simplicity and given the task constraints, we'll ensure data is available in the session state.

