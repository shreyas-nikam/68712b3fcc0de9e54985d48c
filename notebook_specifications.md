
# Technical Specification: Scenario-ILD Integration Workbench Jupyter Notebook

This document outlines the detailed specification for a Jupyter Notebook designed to explore the integration of expert-based scenario assessments with historical Internal Loss Data (ILD) for operational risk modeling. It defines the logical flow, required markdown explanations, and conceptual code sections without implementing actual Python code.

---

## 1. Notebook Overview

This notebook provides an interactive environment to understand and apply techniques for combining distinct operational risk data sources.

### 1.1 Learning Goals
Upon completing this notebook, users will be able to:
*   Understand the key insights contained in the Risk Modeling Section of the attached document [1].
*   Learn the different formats for scenario assessment and their quantification [1, p. 27].
*   Explore methods for fitting distributions to scenario assessment data [1, p. 28-29].
*   Comprehend the techniques for combining ILD and scenario models, including parameter and quantile averaging [1, p. 30-33].
*   Observe the 'stability paradox' where improving 'body' risk might unexpectedly increase overall tail risk [1, p. 30].

### 1.2 Expected Outcomes
Users will gain a hands-on understanding of how subjective expert judgments are quantified into statistical distributions and how various averaging methods influence the final operational risk profile. The interactive demonstrations, particularly the stability paradox, will highlight critical considerations in risk management and model interpretation, reinforcing the importance of precise Unit of Measure (UoM) definition. Users will be able to:
*   Define hypothetical operational risk scenarios with quantified frequency and severity.
*   Generate synthetic historical Internal Loss Data (ILD) reflecting specific risk characteristics.
*   Fit appropriate statistical distributions to both scenario assessments and ILD.
*   Apply and compare different methodologies for combining ILD and scenario-based models.
*   Visualize the impact of different combination strategies on the overall risk distribution.
*   Observe and analyze the 'stability paradox' through controlled simulations.

---

## 2. Mathematical and Theoretical Foundations

This section details the theoretical underpinnings and mathematical formulas used in the notebook, all presented using LaTeX for clarity.

### 2.1 Introduction to Operational Risk Modeling Data Sources
*   **Markdown Explanation (Cell 1)**: Introduction to the four key data categories for operational risk: Internal Loss Data (ILD), External Loss Data (ELD), Scenario Data (SD), and Business Environment and Internal Control Factors (BEICF). Emphasize their complementary nature and the challenge of integration, especially for tail losses.
    *   *Real-world Application*: Discuss why ILD alone is often insufficient for capital modeling, particularly for severe, rare events, necessitating scenario data.

### 2.2 Frequency Modeling
*   **Markdown Explanation (Cell 2)**: Explanation of frequency distributions used for modeling the number of losses ($N$) over a period (typically one year). Introduce the Q-factor as a measure of dispersion.
    *   *Definitions*:
        *   **Q-factor**: The ratio of the variance to the mean of the loss frequency distribution.
            $$Q = \frac{\text{Var}[N]}{E[N]}$$
        *   **Poisson Distribution**: Used when loss events are assumed independent. Characterized by $\lambda$. $Q=1$.
        *   **Negative Binomial Distribution**: Used to introduce dependence among events (higher variance than mean). $Q > 1$.
        *   **Binomial Distribution**: Used for a fixed number of trials (lower variance than mean). $Q < 1$.
    *   *Formulas*:
        *   **Poisson Probability Mass Function (PMF)**:
            $$p(N=n) = \frac{e^{-\lambda}\lambda^n}{n!}$$
            Mean: $E[N] = \lambda$, Variance: $\text{Var}[N] = \lambda$
        *   **Negative Binomial PMF**:
            $$p(N=n) = \frac{\Gamma(\alpha+n)}{\Gamma(\alpha)n!} \left(\frac{1}{1+\beta}\right)^\alpha \left(\frac{\beta}{1+\beta}\right)^n$$
            Mean: $E[N] = \alpha\beta$, Variance: $\text{Var}[N] = \alpha\beta(1+\beta)$
        *   **Binomial PMF**:
            $$p(N=n) = \binom{m}{n} p^n (1-p)^{m-n}$$
            Mean: $E[N] = mp$, Variance: $\text{Var}[N] = mp(1-p)$
    *   *Derivation/Application*: Briefly explain how mean and variance are used to calibrate parameters for these distributions.

### 2.3 Severity Modeling and Distribution Fitting
*   **Markdown Explanation (Cell 3)**: Discuss the challenges of severity fitting, especially for heavy-tailed operational losses. Introduce concepts of risk functionals and empirical risk minimization.
    *   *Definitions*:
        *   **Risk Functional**: A penalty function that quantifies the deviation of a model distribution from the true risk severity distribution. E.g., Kolmogorov-Smirnov distance, log-likelihood.
        *   **Empirical Risk Minimization**: The process of finding the model distribution that best approximates the true risk distribution based on observed data.
        *   **Heavy-tailed Distributions**: Distributions assigning higher probabilities to very large losses, crucial for operational risk. Examples: Generalized Pareto, Pareto, Log-normal.
        *   **Body-tail Distributions**: Composite distributions combining a 'body' part (for smaller losses) and a 'tail' part (for larger, extreme losses) often above a threshold $T$.
    *   *Formulas*:
        *   **Kolmogorov-Smirnov Distance** (for comparing distributions $F_i$ and $F_j$ with $n_i$ and $n_j$ losses respectively):
            $$d_{ij} = \frac{n_i n_j}{n_i+n_j} \sup_x |F_i(x) - F_j(x)|$$
        *   **Body-tail Cumulative Distribution Function (CDF)**:
            $$F_T(x) = \begin{cases} p + (1-p)F_{\text{Tail}}(x) & x \ge T \\ p F_{\text{Body}}(x) & x < T \end{cases}$$
            where $p$ is the proportion of body losses.
        *   **Truncated Distribution Function**:
            $$\text{Tr}[F(x); T] = \begin{cases} \frac{F(x)-F(T)}{1-F(T)} & x \ge T \\ 0 & x < T \end{cases}$$
    *   *Real-world Application*: Emphasize the importance of accurate tail modeling for capital estimation. Discuss fitting methods like truncated fitting and fitting in excess of a threshold.

### 2.4 Combining ILD and Scenario Models
*   **Markdown Explanation (Cell 4)**: Explain the necessity and methodologies for combining ILD and scenario models, especially when they cover the same risks. Highlight the concept of weighting based on model 'precision' or 'data richness'.
    *   *Definitions*:
        *   **Parameter Averaging**: Combining distribution parameters directly (e.g., tail index for Pareto distributions). Applicable when both sources assume the same distribution family.
        *   **Quantile Averaging**: Combining quantiles of the individual distributions at various probability levels. More general, applicable across different distribution families.
        *   **Precision (n, m)**: Represent the effective number of losses or data points contributing to the ILD and scenario models, respectively.
    *   *Formulas*:
        *   **Parameter Averaging (for Pareto tails, $\hat{\xi}_1$ for ILD, $\hat{\xi}_2$ for Scenario)**:
            $$\hat{\xi}_{1,2} = \frac{n}{n+m}\hat{\xi}_1 + \frac{m}{n+m}\hat{\xi}_2$$
        *   **Quantile Averaging with Constant Weights ($\frac{n}{n+m}$ for ILD, $\frac{m}{n+m}$ for Scenario)**:
            $$Q_{1,2}(q) = Q_1(q)^{\frac{n}{n+m}} Q_2(q)^{\frac{m}{n+m}}$$
        *   **Quantile Averaging with Non-Constant Weights** (from feature list, using $F^{-1}$ notation):
            $$F_{IS}^{-1}(q) = \left(F_I^{-1}(q)\right)^{\frac{\omega_S(q)}{\omega_I(q)+\omega_S(q)}} \left(F_S^{-1}(q)\right)^{\frac{\omega_I(q)}{\omega_I(q)+\omega_S(q)}}$$
            where $\omega_I(q)$ and $\omega_S(q)$ are weights based on quantile estimator variances, often estimated via bootstrapping. For Pareto distributions, the variance of the tail parameter estimate $\hat{\xi}$ is proportional to $\xi^2$ divided by the number of observations, i.e., $\text{Var}[\hat{\xi}] \approx \xi^2/N$. Thus, for our purposes:
            $$\omega_I(q) \propto \frac{1}{n} \quad \text{and} \quad \omega_S(q) \propto \frac{1}{m}$$
            (Simplified proportional weights based on precision for demonstration in the notebook.)
        *   **Combined Frequency ($\lambda_I$ for ILD, $\lambda_S$ for Scenario)**:
            $$\lambda_{IS} = \frac{n_I \lambda_I + n_S \lambda_S}{n_I+n_S} = \frac{n_I}{n_I+n_S}\lambda_I + \frac{n_S}{n_I+n_S}\lambda_S$$
    *   *Real-world Application*: Discuss the practical implications of each method, considering data availability and model complexity.

### 2.5 The Stability Paradox
*   **Markdown Explanation (Cell 5)**: Explain the counter-intuitive 'stability paradox' in operational risk modeling.
    *   *Definition*: The phenomenon where reducing losses in the 'body' (smaller, more frequent losses) of a risk distribution, while keeping the 'tail' (large, rare losses) constant, can lead to a *heavier-tailed* overall distribution and potentially higher implied capital.
    *   *Real-world Application*: Emphasize why understanding this paradox is crucial for risk managers to avoid misinterpreting model results and for defining appropriate Units of Measure (UoMs).

---

## 3. Code Requirements

This section details the expected libraries, inputs, outputs, algorithms, and visualizations for the notebook. No actual Python code will be provided, only conceptual descriptions.

### 3.1 Expected Libraries
The notebook will utilize standard open-source Python libraries available on PyPI:
*   `numpy`: For numerical operations, array manipulation, and statistical calculations.
*   `pandas`: For data manipulation and tabular data handling.
*   `scipy.stats`: For statistical distributions (e.g., Pareto, Generalized Pareto, Log-Normal, Poisson, Negative Binomial) and fitting functions.
*   `matplotlib.pyplot`: For static plotting and visualization.
*   `seaborn`: For enhanced statistical data visualization.
*   `ipywidgets`: For creating interactive user controls (sliders, dropdowns, text inputs).
*   `scipy.optimize`: For numerical optimization routines (e.g., least squares fitting).

### 3.2 Input/Output Expectations

#### 3.2.1 Inputs (User-Defined Parameters)
The notebook will feature interactive elements (`ipywidgets`) to allow users to modify parameters and rerun analyses. Inline help text or tooltips will be provided for each control.

*   **Scenario Definition Parameters**:
    *   `scenario_expected_frequency`: Text input/slider for expected annual frequency of the scenario (e.g., "1 loss per year").
    *   `scenario_50_percentile_loss`: Text input/slider for 50th percentile loss amount.
    *   `scenario_90_percentile_loss`: Text input/slider for 90th percentile loss amount.
    *   `scenario_99_percentile_loss`: Text input/slider for 99th percentile loss amount.
    *   `scenario_precision_m`: Text input/slider for 'precision' of scenario data ($m$, representing equivalent number of observations).
*   **Synthetic ILD Generation Parameters**:
    *   `ild_frequency_mean`: Text input/slider for mean annual frequency of ILD (e.g., "100 losses per year").
    *   `ild_frequency_dispersion`: Text input/slider for dispersion/over-dispersion parameter (e.g., for Negative Binomial, or a switch for Poisson vs. Negative Binomial).
    *   `ild_severity_distribution`: Dropdown for chosen severity distribution (e.g., "Log-Normal", "Generalized Pareto", "Pareto").
    *   `ild_severity_params`: Text inputs/sliders for distribution parameters (e.g., mean, std dev for Log-Normal; shape, scale for Pareto/GPD).
    *   `num_ild_observations`: Text input/slider for the number of synthetic ILD observations to generate ($n$). This will also be used as the 'precision' for ILD.
    *   `ild_reporting_threshold`: Text input/slider for a minimum loss amount for ILD reporting.
*   **Combination Method Selection**:
    *   `combination_method`: Dropdown for "Parameter Averaging", "Quantile Averaging (Constant Weights)", "Quantile Averaging (Non-Constant Weights)".
*   **Stability Paradox Controls**:
    *   `paradox_body_reduction_factor`: Slider to reduce the frequency/severity of smaller (body) losses for the scenario.
    *   `paradox_tail_fixed_percentile`: Text input for the percentile at which the tail of the scenario distribution should remain fixed.

#### 3.2.2 Outputs (Generated Results)
*   **Text/Table Outputs**:
    *   Summary of input parameters.
    *   Fitted parameters for scenario distribution.
    *   Summary statistics for generated synthetic ILD.
    *   Fitted parameters for ILD distribution.
    *   Calculated combined distribution parameters or key quantiles.
    *   Estimated Capital (e.g., 99.9% Value at Risk (VaR), Expected Shortfall (ES)) for individual and combined distributions.
    *   Comparison table of capital estimates for different combination methods.
*   **Visual Outputs**: Plots and charts as detailed in section 3.4.

### 3.3 Algorithms and Functions (Conceptual)

The following functions will be implemented within the notebook's code cells. No Python code is to be included here, only conceptual descriptions.

*   **`fit_scenario_distribution(frequency, percentile_50, percentile_90, percentile_99)`**
    *   *Description*: Takes scenario frequency and severity quantiles as input. It will fit a Generalized Pareto Distribution (GPD) or another suitable heavy-tailed distribution to these points using a least squares or maximum likelihood approach on the quantile function. This simulates translating expert judgment into a statistical distribution.
    *   *Output*: Fitted distribution object (e.g., `scipy.stats.genpareto` instance) and its parameters.

*   **`generate_synthetic_ild(frequency_params, severity_params, num_observations, reporting_threshold)`**
    *   *Description*: Generates a synthetic dataset of operational losses.
        *   Simulates annual loss frequencies based on `frequency_params` (e.g., Poisson or Negative Binomial).
        *   For each loss event, generates a severity amount based on `severity_params` (e.g., Log-Normal, Pareto, GPD).
        *   Filters losses below the `reporting_threshold`.
    *   *Output*: A Pandas DataFrame containing `Loss_ID`, `Amount`, `Date` (optional, for time-series context if applicable).

*   **`fit_ild_distribution(ild_data, distribution_type, threshold)`**
    *   *Description*: Fits a distribution to the generated synthetic ILD. This may involve fitting a body-tail distribution (e.g., empirical for body, GPD for tail above `threshold`) or a single parametric distribution (e.g., Log-Normal, GPD) depending on `distribution_type`. Uses maximum likelihood estimation for parametric fits.
    *   *Output*: Fitted distribution object and its parameters.

*   **`combine_distributions_param_avg(ild_dist_params, scenario_dist_params, n_ild, m_scenario)`**
    *   *Description*: If both ILD and scenario severities are approximated by Pareto-like tails, this function calculates a combined tail parameter using the weighted average formula provided in section 2.4.
    *   *Output*: Combined distribution parameters.

*   **`combine_distributions_quantile_avg_constant_weights(ild_quantile_func, scenario_quantile_func, n_ild, m_scenario, quantiles_to_evaluate)`**
    *   *Description*: Combines the quantile functions of the ILD and scenario models using the geometric average with constant weights proportional to `n_ild` and `m_scenario`.
    *   *Output*: A new quantile function or a set of combined quantiles across `quantiles_to_evaluate`.

*   **`combine_distributions_quantile_avg_variable_weights(ild_quantile_func, scenario_quantile_func, n_ild, m_scenario, quantiles_to_evaluate, num_bootstraps)`**
    *   *Description*: Implements quantile averaging with non-constant weights. This involves:
        1.  Parametric bootstrapping on the individual ILD and scenario fits to estimate the variance of their quantile estimators.
        2.  Calculating quantile-specific weights based on these variances.
        3.  Combining quantiles using the variable-weighted geometric average formula.
    *   *Output*: A new quantile function or a set of combined quantiles.

*   **`calculate_capital(combined_distribution_or_quantiles, confidence_level)`**
    *   *Description*: Estimates risk capital (e.g., 99.9% VaR, Expected Shortfall) from a given distribution or a set of quantiles. This will likely involve Monte Carlo simulation of aggregate losses if frequency is also considered, or direct quantile lookup for severity-only analysis.
    *   *Output*: Numeric value for VaR and/or ES.

*   **`simulate_stability_paradox(base_scenario_params, modified_body_scenario_params, ild_params)`**
    *   *Description*: Simulates two scenarios: a base case and one with improved 'body' risk (e.g., lower frequency/severity of small losses) but constant 'tail' risk. It then fits/combines these with ILD and calculates capital to demonstrate the paradox.
    *   *Output*: Capital estimates and distributions for both cases, highlighting the counter-intuitive increase in tail risk.

### 3.4 Visualizations

Visualizations will adhere to best practices: color-blind-friendly palettes, font size $\ge 12$ pt, clear titles, labeled axes, and legends. Interactivity will be enabled where supported by the environment, with static (PNG) fallbacks.

*   **Core Visuals**:
    1.  **Distribution Comparison Plot (CDF/PDF)**:
        *   *Type*: Line plot (CDF) or Area/Line plot (PDF).
        *   *Content*: Display the Cumulative Distribution Functions (CDFs) and/or Probability Density Functions (PDFs) for:
            *   Fitted Scenario Severity Distribution.
            *   Fitted ILD Severity Distribution.
            *   Combined Severity Distribution (for each selected combination method).
        *   *Purpose*: Visually compare the individual risk profiles and the effect of different aggregation approaches.
        *   *Interactive Elements*: Dropdown to select different combination methods for display.

    2.  **Synthetic ILD Data Analysis Plots**:
        *   *Type*: Histogram and/or Box Plot.
        *   *Content*: Distribution of generated synthetic ILD amounts.
        *   *Purpose*: Verify the realism of the generated data and observe its characteristics.
        *   *Interactive Elements*: Sliders to adjust bin count for histogram.

    3.  **Quantile Comparison Plot**:
        *   *Type*: Scatter plot with lines.
        *   *Content*: Plot of quantiles (x-axis, e.g., 90th, 95th, 99th, 99.9th percentile) vs. loss amounts (y-axis) for:
            *   Fitted Scenario Distribution.
            *   Fitted ILD Distribution.
            *   Combined Distribution (for each method).
        *   *Purpose*: Allow for direct comparison of risk levels at various confidence points.
        *   *Interactive Elements*: Checkboxes to toggle visibility of individual distributions.

    4.  **Stability Paradox Demonstration Plot**:
        *   *Type*: Line plot (CDF) or Overlayed PDF plots.
        *   *Content*: Display the combined severity distributions for:
            *   Base Case (original scenario + ILD).
            *   Paradox Case (modified scenario body + ILD).
        *   *Purpose*: Visually demonstrate how a change in the 'body' of one distribution can lead to a heavier tail in the combined distribution, despite the tail of the individual distribution remaining constant. Highlight the VaR or ES points.

*   **Tables**:
    *   Summary tables for fitted distribution parameters.
    *   Table comparing VaR and ES estimates for individual and combined models across different combination methods and for the stability paradox demonstration.

---

## 4. Additional Notes or Instructions

### 4.1 Assumptions
*   The provided document [1] serves as the primary theoretical reference. Users are assumed to have basic familiarity with statistical distributions (e.g., PDF, CDF, quantiles) and risk modeling concepts (VaR, ES).
*   For simplicity and clarity, the notebook will focus on severity modeling and combination, assuming frequency models are either given or integrated via simulation (e.g., Compound Poisson).
*   Synthetic data generation aims for realism but is not calibrated to specific real-world datasets. The chosen distributions and parameters are for illustrative purposes.
*   Precision parameters ($n$ and $m$) for ILD and scenario data are user-defined and represent conceptual "equivalent observations" for weighting purposes, consistent with the handbook.

### 4.2 Constraints
*   **Performance**: The entire notebook must execute end-to-end on a mid-spec laptop (8 GB RAM) in fewer than 5 minutes. This implies efficient algorithms and careful selection of simulation sizes.
*   **Open-Source Only**: All dependencies must be open-source Python libraries from PyPI. No proprietary software or specific platform features (like Streamlit) are to be used or referenced.
*   **Narrative Clarity**: All major analytical and modeling steps will be accompanied by:
    *   Brief narrative markdown cells explaining `what` is being done and `why` it is important in the context of operational risk modeling.
    *   Descriptive code comments within code cells.
*   **Synthetic Dataset Requirements**:
    *   **Content**: Generated synthetic data will include realistic numeric loss amounts. Categorical (e.g., event type, business unit) and time-series (e.g., loss date) fields will be included to provide context, even if not fully leveraged in all specific analyses, to make the data feel realistic.
    *   **Handling & Validation**: Initial code cells will include checks for expected column names, data types, and primary-key uniqueness (if applicable). Assertions will confirm no missing values in critical fields, and summary statistics for numeric columns will be logged.
    *   **Sample Data**: An optional, lightweight (e.g., $\le$ 5 MB) sample of synthetic data will be pre-generated and available (e.g., as a CSV file) so the notebook can run even if the user skips the synthetic data generation step or if interactive generation fails.

### 4.3 Customization Instructions
*   **Parameter Adjustments**: Users are encouraged to experiment with the provided sliders and input fields for scenario definition, ILD generation, and precision parameters to observe their impact on the fitted distributions, combined models, and capital estimates.
*   **Distribution Choices**: While the core examples will use Generalized Pareto and Log-Normal, instructions will guide users on how to modify the code (e.g., by changing `scipy.stats` calls) to explore other heavy-tailed distributions if desired.
*   **Scenario Exploration**: Users can define multiple hypothetical scenarios by modifying the quantile inputs to understand how different expert judgments translate into risk profiles.

---

### References
*   [1] Jonathan Howitt (Editor), *PRMIA Operational Risk Manager Handbook*, The Professional Risk Managers' International Association, Updated November 2015.
*   `numpy`
*   `pandas`
*   `scipy`
*   `matplotlib`
*   `seaborn`
*   `ipywidgets`

