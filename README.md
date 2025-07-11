# QuLab: Operational Risk Modeling & Aggregation Sandbox

## Project Title

**QuLab: Interactive Operational Risk Modeling and Aggregation Lab**

## Project Description

QuLab is a Streamlit-powered interactive application designed for students and professionals to explore fundamental concepts in operational risk modeling and aggregation. It bridges the gap between theoretical understanding and practical application by providing a hands-on environment to:

*   Define and fit statistical distributions to hypothetical operational risk scenarios.
*   Generate and fit distributions to synthetic Internal Loss Data (ILD).
*   Combine different risk distributions using statistical averaging techniques.
*   Visually demonstrate the counter-intuitive "stability paradox" in risk aggregation, where seemingly positive changes in the body of a loss distribution can paradoxically increase overall tail risk.

This lab project aims to deepen understanding of quantitative risk management techniques through interactive simulations and visualizations.

## Features

QuLab is structured into three main interactive sections, accessible via the sidebar navigation:

1.  ### **Scenario Distribution Fitting**
    *   Define hypothetical operational risk scenarios by inputting expected annual frequency and key loss percentiles (50th, 90th, 99th).
    *   Automatically fits a Log-Normal distribution to these expert assessments.
    *   Visualizes the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) of the fitted scenario distribution.
    *   Provides mathematical formulas for the fitted distribution.

2.  ### **ILD Generation and Fitting**
    *   Generate synthetic Internal Loss Data (ILD) by specifying frequency (Poisson) and severity (Log-Normal, Pareto, GPD) parameters, number of observations, and a reporting threshold.
    *   Display summary statistics and a histogram of the generated ILD.
    *   Fit statistical distributions to the generated ILD, offering two options:
        *   **Log-Normal Fit:** Fits a single Log-Normal distribution to the entire dataset.
        *   **Body-Tail Fit:** Empirically models the "body" of the distribution and fits a Generalized Pareto Distribution (GPD) to losses exceeding a user-defined "tail" threshold.
    *   Visualizes the fitted distributions against the ILD histogram and ECDF.

3.  ### **Distribution Combination and Paradox Simulation**
    *   Explore methods for combining different risk distributions (e.g., scenario and ILD):
        *   **Parameter Averaging:** Demonstrates combining tail parameters based on precision weights.
        *   **Quantile Averaging with Constant Weights:** Combines quantiles using a geometric average with user-defined precision weights, visualized with quantile plots.
    *   **Stability Paradox Simulation:**
        *   Set parameters for a "base" scenario, a "modified body" scenario (where the body might seem "safer"), and a set of ILD.
        *   Simulates aggregate losses for both base and modified scenarios combined with ILD.
        *   Calculates and compares Value at Risk (VaR) for the combined distributions, highlighting when the stability paradox occurs.
        *   Visualizes the loss distributions to intuitively understand the paradox.

## Getting Started

Follow these instructions to set up and run the QuLab application on your local machine.

### Prerequisites

*   Python 3.8+ (recommended)
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd qu_lab_project # Or whatever your project folder is named
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    Create a `requirements.txt` file in the root directory of your project (same level as `app.py`) with the following content:

    ```
    streamlit
    scipy
    numpy
    pandas
    plotly
    ```

    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is active and you are in the root directory of the project (where `app.py` is located).
    ```bash
    streamlit run app.py
    ```
2.  **Access the application:**
    Your web browser will automatically open a new tab with the Streamlit application (usually at `http://localhost:8501`).

3.  **Navigate and Interact:**
    *   Use the **sidebar** on the left to navigate between the "Scenario Fitting", "ILD Generation and Fitting", and "Distribution Combination and Paradox Simulation" pages.
    *   Adjust input parameters using sliders, number inputs, and selectboxes.
    *   Observe the real-time updates in plots and calculated values.
    *   Read the explanatory text on each page to understand the concepts being demonstrated.

## Project Structure

```
.
├── app.py
├── application_pages/
│   ├── __init__.py
│   ├── page1.py
│   ├── page2.py
│   └── page3.py
└── requirements.txt
```

*   `app.py`: The main Streamlit application file. It sets up the page configuration, displays the main title, and handles navigation between different pages.
*   `application_pages/`: A directory containing separate Python modules for each distinct page or section of the application.
    *   `page1.py`: Contains the logic for the "Scenario Fitting" functionality.
    *   `page2.py`: Contains the logic for "ILD Generation and Fitting".
    *   `page3.py`: Contains the logic for "Distribution Combination and Paradox Simulation".
*   `requirements.txt`: Lists all Python dependencies required to run the application.

## Technology Stack

*   **Python 3.x**: The core programming language.
*   **Streamlit**: For building the interactive web application interface.
*   **SciPy**: For statistical distributions, probability functions, and distribution fitting algorithms.
*   **NumPy**: For numerical operations and array manipulation.
*   **Pandas**: For data manipulation and analysis, particularly for the synthetic ILD.
*   **Plotly**: For creating interactive and dynamic data visualizations (histograms, PDFs, CDFs, scatter plots).
*   **Geometric Mean and Quantile Averaging**: Specific statistical techniques used for distribution combination.
*   **Caching (`@st.cache_data`)**: Used for optimizing performance of data generation and fitting functions in Streamlit.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

*   Fork the repository.
*   Create a new branch for your feature or bug fix.
*   Submit a pull request with your changes.
*   Report bugs or suggest enhancements via the GitHub Issues page.

## License

This project is licensed under the MIT License - see the `LICENSE` file (if you create one) for details. For a lab project, this usually implies free use and modification.

## Contact

For any questions or further information, please contact:

*   **Quant University**
*   **Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   *(You might also add specific author names/emails if this is a personal project within a lab.)*
