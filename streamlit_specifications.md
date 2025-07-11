
# Streamlit Application Requirements Specification: Operational Risk Modeling Workbench

This document outlines the requirements for developing an interactive Streamlit application based on the provided Jupyter Notebook content. It details the purpose, user interface, and integration of the core functionalities from the notebook.

## 1. Application Overview

**Purpose and Objectives:**
The primary purpose of this Streamlit application is to provide an interactive workbench for exploring operational risk modeling concepts. It aims to bridge the gap between theoretical understanding and practical application by allowing users to define hypothetical scenarios, simulate Internal Loss Data (ILD), and combine these data sources using various statistical averaging techniques. A key objective is to visually demonstrate the counter-intuitive 'stability paradox' in risk aggregation.

**Learning Outcomes Supported:**
*   Understanding of key insights from the "Risk Modeling Section" of the PRMIA Operational Risk Manager Handbook.
*   Familiarity with different formats for scenario assessment and their quantification.
*   Methods for fitting statistical distributions to scenario assessment data.
*   Techniques for combining ILD and scenario models, including parameter and quantile averaging.
*   Observation and comprehension of the 'stability paradox' and its implications for operational risk management.

## 2. User Interface Requirements

**Layout and Navigation Structure:**
The application will adopt a clean, intuitive layout with a sidebar for user inputs and a main area for displaying results and visualizations.
*   **Sidebar:** Will host all interactive input widgets, organized into logical sections corresponding to the core functionalities (e.g., "Scenario Definition," "ILD Generation & Fitting," "Distribution Combination Methods," "Stability Paradox Simulation").
*   **Main Area:** Dedicated to presenting:
    *   Narrative explanations for each step.
    *   Tables of generated data and calculated parameters.
    *   Interactive plots (histograms, PDFs, CDFs, quantile plots) for visualizing distributions and comparison.
    *   Summary statistics and key insights.
*   **Navigation:** Users will interact primarily through the sidebar controls, with results updating dynamically in the main area. Clear section headers and expanders will organize content for readability.

**Input Widgets and Controls:**
The application will utilize a variety of Streamlit widgets to enable user interaction:
*   `st.number_input`: For numerical parameters such as frequencies, percentiles, precision values ($n$, $m$), reporting thresholds, and distribution parameters (e.g., mean, standard deviation, shape, scale, location).
*   `st.slider`: For continuous numerical ranges where a visual slider is beneficial (e.g., number of observations, confidence levels).
*   `st.selectbox`: For selecting categorical options like distribution types ('lognorm', 'pareto', 'gpd', 'bodytail') or combination methods.
*   `st.multiselect`: For selecting multiple quantiles to evaluate.
*   `st.expander`: To organize complex input sections or hide less frequently used options.

**Visualization Components:**
*   **Distribution Plots:**
    *   **Histograms:** To visualize the empirical distribution of generated synthetic ILD.
    *   **Probability Density Functions (PDFs) / Cumulative Distribution Functions (CDFs):** To display fitted scenario, ILD, and combined distributions.
    *   **Quantile Plots:** To compare quantile functions of individual and combined distributions, especially for quantile averaging methods.
*   **Comparative Charts:**
    *   **Bar Charts/Tables:** To compare calculated VaR values in the stability paradox demonstration.
    *   **Overlaid Plots:** To visually compare distributions before and after combination, or between base and modified scenarios in the paradox.
*   **Usability:** All plots will feature clear titles, labeled axes, and legends. A color-blind-friendly palette will be adopted, and font sizes will be $\ge 12$ pt as per requirements.

**Interactive Elements and Feedback Mechanisms:**
*   **Dynamic Updates:** Most outputs will update automatically as input parameters are changed.
*   **Inline Help/Tooltips:** The `help` argument will be used for `st` widgets to provide concise explanations for each control, aligning with user requirements.
*   **Result Display:** Calculated values, parameters, and summary statistics will be displayed clearly using `st.write` or `st.dataframe`.
*   **Error Handling:** Informative messages will be displayed for invalid inputs (e.g., threshold too high for `bodytail` fit).

## 3. Additional Requirements

**Real-time Updates and Responsiveness:**
Streamlit's inherent design supports real-time updates. The application will re-render outputs dynamically whenever an input widget's value changes, ensuring a responsive user experience. For computationally intensive operations (e.g., extensive bootstrapping or simulations), `@st.cache_data` will be utilized to optimize performance and prevent unnecessary re-runs, ensuring the application executes efficiently within the specified time limits.

**Annotation and Tooltip Specifications:**
Every input widget and significant output will include descriptive inline help text or tooltips. This will be implemented using the `help` parameter available in Streamlit widgets (e.g., `st.slider(..., help="Descriptive text here")`) to guide users on the purpose and effect of each control and data point.

## 4. Notebook Content and Code Requirements

This section details how the core functionalities and code from the Jupyter Notebook will be integrated into the Streamlit application.

### 4.1. Scenario Distribution Fitting

**Purpose:** This component translates expert-based scenario assessments (frequency and impact quantiles) into a statistical distribution, specifically a Log-Normal distribution in this implementation. This is crucial for quantifying qualitative risk insights.

**Inputs from UI:**
*   `Scenario Frequency`: Expected annual frequency of the scenario (e.g., 1 loss per year).
    *   **Widget**: `st.number_input("Expected Annual Frequency (Scenario)", value=1, min_value=0.01, help="Expected number of loss events per year for the scenario.")`
*   `Scenario 50th Percentile`: The 50th percentile loss amount for the scenario.
    *   **Widget**: `st.number_input("50th Percentile Loss (Scenario)", value=1_000_000, min_value=1.0, help="The loss value below which 50% of scenario losses fall.")`
*   `Scenario 90th Percentile`: The 90th percentile loss amount for the scenario.
    *   **Widget**: `st.number_input("90th Percentile Loss (Scenario)", value=5_000_000, min_value=1.0, help="The loss value below which 90% of scenario losses fall.")`
*   `Scenario 99th Percentile`: The 99th percentile loss amount for the scenario.
    *   **Widget**: `st.number_input("99th Percentile Loss (Scenario)", value=10_000_000, min_value=1.0, help="The loss value below which 99% of scenario losses fall.")`

**Outputs to UI:**
*   Display of the fitted distribution type and its estimated parameters (shape, location, scale).
*   A plot of the Probability Density Function (PDF) or Cumulative Distribution Function (CDF) of the fitted scenario distribution.

**Relevant Code (from Jupyter Notebook):**
```python
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
```

### 4.2. Synthetic ILD Generation

**Purpose:** This component generates synthetic Internal Loss Data (ILD), simulating historical operational losses. This dataset serves as a controllable input for subsequent modeling and analysis steps, particularly useful for testing different modeling approaches.

**Inputs from UI:**
*   `Frequency Distribution (ILD)`: Type of frequency distribution (e.g., Poisson).
    *   **Widget**: `st.selectbox("ILD Frequency Distribution", ["poisson"], help="Choose the distribution for the number of losses per period.")`
*   `Frequency Lambda (ILD)`: The `lambda` parameter for the Poisson frequency distribution (average number of events).
    *   **Widget**: `st.number_input("ILD Frequency Rate (lambda)", value=100, min_value=1, help="Average number of loss events per period (for Poisson).")`
*   `Severity Distribution (ILD)`: Type of severity distribution (e.g., Log-Normal, Pareto, GPD).
    *   **Widget**: `st.selectbox("ILD Severity Distribution", ["lognorm", "pareto", "gpd"], help="Choose the distribution for individual loss amounts.")`
*   `Severity Parameters (ILD)`: Parameters specific to the chosen severity distribution.
    *   **Widgets (conditional based on selection)**:
        *   For `lognorm`: `st.number_input("Mean (lognorm)", value=10000, help="Geometric mean for log-normal severity.")`, `st.number_input("Sigma (lognorm)", value=0.5, help="Standard deviation of log-transformed data for log-normal severity.")`
        *   For `pareto`: `st.number_input("Shape (Pareto)", value=1.0, help="Shape parameter 'b' for Pareto severity.")`, `st.number_input("Location (Pareto)", value=0.0, help="Location parameter for Pareto severity.")`, `st.number_input("Scale (Pareto)", value=1.0, help="Scale parameter for Pareto severity.")`
        *   For `gpd`: `st.number_input("Shape (GPD)", value=0.1, help="Shape parameter 'c' for GPD severity.")`, `st.number_input("Location (GPD)", value=0.0, help="Location parameter for GPD severity.")`, `st.number_input("Scale (GPD)", value=1.0, help="Scale parameter for GPD severity.")`
*   `Number of Observations`: Number of periods to simulate ILD.
    *   **Widget**: `st.slider("Number of Simulation Periods (ILD)", value=5, min_value=1, max_value=20, help="Number of periods (e.g., years) to simulate ILD for.")`
*   `Reporting Threshold`: Minimum loss amount to be recorded in the synthetic ILD.
    *   **Widget**: `st.number_input("ILD Reporting Threshold", value=1000, min_value=0.0, help="Only losses above this amount are recorded in ILD.")`

**Outputs to UI:**
*   A DataFrame displaying the first few rows of the generated ILD.
*   Summary statistics (total losses, mean loss amount, max loss amount).
*   A histogram of the generated ILD `Amount` to visualize its empirical distribution.

**Relevant Code (from Jupyter Notebook):**
```python
import scipy.stats as stats
import numpy as np
import pandas as pd
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
```

### 4.3. ILD Distribution Fitting

**Purpose:** This function fits a statistical distribution to the generated ILD, allowing for generalization and inference about future loss events. It supports fitting a Log-Normal distribution or a Body-Tail distribution (empirical body, Generalized Pareto Distribution (GPD) tail).

**Inputs from UI:**
*   `ILD Data`: The `Amount` column from the `ild_data` DataFrame generated in the previous step (passed internally).
*   `Distribution Type (ILD Fit)`: Choice between 'lognorm' and 'bodytail'.
    *   **Widget**: `st.selectbox("ILD Distribution Fit Type", ["lognorm", "bodytail"], help="Choose 'lognorm' for a single distribution fit, or 'bodytail' for a combined empirical/GPD fit.")`
*   `Threshold for Body-Tail Fit`: The value separating the 'body' (smaller losses) from the 'tail' (larger losses) when `distribution_type` is 'bodytail'.
    *   **Widget (conditional)**: `st.number_input("Threshold for Body-Tail Fit", value=10000, min_value=0.0, help="Losses above this threshold are modeled by GPD (for 'bodytail' fit type).")`

**Outputs to UI:**
*   Display of the fitted distribution type(s) and their parameters.
*   A plot showing the fitted distribution's PDF/CDF, ideally overlaid with a histogram of the ILD data for visual goodness-of-fit assessment.

**Relevant Code (from Jupyter Notebook):**
```python
import scipy.stats as stats
import numpy as np

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
```
**Mathematical Context for GPD Tail:**
The Generalized Pareto Distribution (GPD) is defined by its cumulative distribution function (CDF) for excesses $x$ over a threshold $u$:
$$G_{\xi,\sigma}(x) = 1 - \left(1 + \frac{\xi (x-u)}{\sigma}\right)^{-1/\xi}$$
for $x > u$, where:
*   $\xi$ is the shape parameter (or tail index), governing the heaviness of the tail.
*   $\sigma$ is the scale parameter, related to the dispersion of data beyond the threshold.
*   $u$ is the threshold.
If $\xi = 0$, the GPD becomes an exponential distribution:
$$G_{0,\sigma}(x) = 1 - e^{-(x-u)/\sigma}$$

### 4.4. Combining Distributions - Parameter Averaging

**Purpose:** This method combines distributions by averaging their parameters, particularly relevant when both ILD and scenario severities are approximated by Pareto-like tails. It uses a weighted average based on the precision (number of losses) of each source.

**Inputs from UI:**
*   `ILD Tail Parameter`: The estimated tail parameter ($\hat{\xi}_1$) from the ILD distribution (e.g., from a GPD fit).
    *   **Widget**: `st.number_input("ILD Tail Parameter (ξ1)", value=0.5, min_value=0.0, help="Shape parameter for ILD's Pareto-like tail.")`
*   `Scenario Tail Parameter`: The estimated tail parameter ($\hat{\xi}_2$) from the scenario distribution.
    *   **Widget**: `st.number_input("Scenario Tail Parameter (ξ2)", value=0.7, min_value=0.0, help="Shape parameter for Scenario's Pareto-like tail.")`
*   `Number of ILD Losses (n)`: The effective number of losses representing the precision of the ILD data.
    *   **Widget**: `st.number_input("ILD Precision (n)", value=100, min_value=1, help="Effective number of losses representing ILD precision.")`
*   `Number of Scenario Losses (m)`: The effective number of losses representing the precision of the scenario data.
    *   **Widget**: `st.number_input("Scenario Precision (m)", value=10, min_value=1, help="Effective number of losses representing scenario precision.")`

**Outputs to UI:**
*   The calculated combined tail parameter ($\hat{\xi}_{1,2}$).
*   Textual explanation of how the weighting impacts the combined parameter.

**Relevant Code (from Jupyter Notebook):**
```python
def combine_distributions_param_avg(ild_dist_params, scenario_dist_params, n_ild, m_scenario):
    """If both ILD and scenario severities are approximated by Pareto-like tails, this function calculates a combined tail parameter using the weighted average formula."""
    return (ild_dist_params * n_ild + scenario_dist_params * m_scenario) / (n_ild + m_scenario)
```
**Mathematical Context:**
The combined tail parameter $\hat{\xi}_{1,2}$ is calculated as a weighted average:
$$\hat{\xi}_{1,2} = \frac{\hat{\xi}_1 n + \hat{\xi}_2 m}{n+m}$$

### 4.5. Combining Distributions - Quantile Averaging with Constant Weights

**Purpose:** This method combines the quantiles of individual ILD and scenario distributions using a geometric average with constant weights, which are proportional to the precision of each data source. This is useful when distributions are of different types.

**Inputs from UI:**
*   `ILD Quantile Function`: This will be derived internally from the fitted ILD distribution (`ild_lognorm_dist.ppf` or equivalent).
*   `Scenario Quantile Function`: This will be derived internally from the fitted scenario distribution (`scenario_lognorm_dist.ppf` or equivalent).
*   `Number of ILD Losses (n)`: Precision of ILD. (Same as in Parameter Averaging, can be reused or set explicitly).
*   `Number of Scenario Losses (m)`: Precision of Scenario. (Same as in Parameter Averaging, can be reused or set explicitly).
*   `Quantiles to Evaluate`: List of probabilities (e.g., 0.90, 0.95, 0.99) at which to calculate combined quantiles.
    *   **Widget**: `st.multiselect("Quantiles to Evaluate (Constant Weights)", [0.90, 0.95, 0.99, 0.995], default=[0.90, 0.95, 0.99], help="Select the percentiles for which to calculate combined quantiles.")`

**Outputs to UI:**
*   A table displaying the individual ILD and scenario quantiles, along with the combined quantiles for the selected probabilities.
*   A plot illustrating the quantile functions of ILD, scenario, and the combined distribution.

**Relevant Code (from Jupyter Notebook):**
```python
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
```
**Mathematical Context:**
The combined quantile $Q_{1,2}(q)$ at probability $q$ is calculated using the geometric average:
$$Q_{1,2}(q) = Q_1(q)^{\frac{n}{n+m}} Q_2(q)^{\frac{m}{n+m}}$$

### 4.6. Combining Distributions - Quantile Averaging with Variable Weights

**Purpose:** This advanced method combines quantiles by using weights that dynamically change across the distribution, based on the estimated variance of quantile estimators. This typically involves a bootstrapping process to assess uncertainty.

**Inputs from UI:**
*   `ILD Quantile Function (with noise)`: Internally derived, potentially with added noise for robust bootstrapping.
*   `Scenario Quantile Function (with noise)`: Internally derived, potentially with added noise for robust bootstrapping.
*   `Number of ILD Losses (n)`: Precision of ILD.
*   `Number of Scenario Losses (m)`: Precision of Scenario.
*   `Quantiles to Evaluate`: List of probabilities for quantile calculation.
*   `Number of Bootstraps`: The number of bootstrap samples used to estimate the variance of quantile estimators.
    *   **Widget**: `st.number_input("Number of Bootstrap Samples", value=200, min_value=10, max_value=1000, step=10, help="Number of samples for bootstrapping to estimate quantile variances.")`

**Outputs to UI:**
*   A table displaying the combined quantiles for the selected probabilities.
*   A plot illustrating the quantile functions of ILD, scenario, and the combined distribution, highlighting how weights might vary.

**Relevant Code (from Jupyter Notebook):**
```python
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
            # Fallback to arithmetic mean if quantiles are non-positive, as geometric mean requires positive values
            combined_quantile = (ild_weight * ild_quantile) + (scenario_weight * scenario_quantile)
        else:
            combined_quantile = (ild_quantile**ild_weight) * (scenario_quantile**scenario_weight)

        combined_quantiles.append(combined_quantile)
    return combined_quantiles
```
**Mathematical Context:**
The combined quantile function $F_{IS}^{-1}(q)$ is calculated using variable weights $\omega_I(q)$ and $\omega_S(q)$ derived from inverse variances of quantile estimators:
$$F_{IS}^{-1}(q) = (F_I^{-1}(q))^{\frac{\omega_S(q)}{\omega_I(q)+\omega_S(q)}} (F_S^{-1}(q))^{\frac{\omega_I(q)}{\omega_I(q)+\omega_S(q)}}$$

### 4.7. Capital Calculation

**Purpose:** This component estimates the risk capital, typically representing a high percentile (e.g., Value-at-Risk or VaR) of the aggregated loss distribution.

**Inputs from UI:**
*   `Combined Distribution or Quantiles`: This will be the output (list of quantiles) from one of the combination methods selected by the user.
*   `Confidence Level`: The percentile at which capital is to be calculated (e.g., 0.995 for 99.5% VaR).
    *   **Widget**: `st.slider("Confidence Level for Capital Calculation", min_value=0.90, max_value=0.999, value=0.995, step=0.001, format="%.3f", help="The percentile (e.g., 0.995 for 99.5% VaR) at which to estimate capital.")`

**Outputs to UI:**
*   The estimated risk capital in numerical format.

**Relevant Code (from Jupyter Notebook):**
```python
import numpy as np

def calculate_capital(combined_distribution_or_quantiles, confidence_level):
    """Estimates risk capital (e.g., 99.9% VaR, Expected Shortfall)."""
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
```

### 4.8. Simulating the Stability Paradox

**Purpose:** This section demonstrates the 'stability paradox' where seemingly positive changes in the 'body' of a risk distribution (e.g., reduced frequency of small losses) can paradoxically lead to an increase in overall tail risk and implied capital.

**Inputs from UI:**
*   `Base Scenario Parameters`: Inputs defining the initial scenario (frequency, 50th, 90th, 99th percentiles).
    *   **Widgets (grouped via `st.expander`)**: `st.number_input` for `frequency_mean`, `severity_50`, `severity_90`, `severity_99`.
*   `Modified Body Scenario Parameters`: Inputs defining the scenario with 'improved body' (lower frequency, but same tail percentiles).
    *   **Widgets (grouped via `st.expander`)**: `st.number_input` for `frequency_mean`, `severity_50`, `severity_90`, `severity_99`.
*   `ILD Parameters for Paradox`: Parameters for simulating ILD data to be combined with both scenarios.
    *   **Widgets (grouped via `st.expander`)**: `st.number_input` for `frequency_mean`, `frequency_dispersion`, `severity_params` (mean, std), `num_observations`, `reporting_threshold`.

**Outputs to UI:**
*   Calculated VaR (e.g., 99.5% VaR) for both the combined base losses and combined modified losses.
*   A clear statement indicating whether the stability paradox was observed (`Modified Scenario VaR` > `Base Scenario VaR`).
*   Visual comparison (e.g., overlaid histograms or CDFs) of the aggregate loss distributions for the base and modified scenarios, emphasizing the tail behavior.

**Relevant Code (from Jupyter Notebook):**
```python
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
```
