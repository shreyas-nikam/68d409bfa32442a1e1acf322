This document outlines the Streamlit application designed for an interactive exploration of phoneme characteristics, their role in Text-to-Speech (TTS) technology, and potential applications in language learning.

---

# Phoneme Characteristics Analysis: An Interactive Exploration for Text-to-Speech Understanding

## Project Title

**Phoneme Characteristics Analysis: An Interactive Exploration for Text-to-Speech Understanding**

## Project Description

This Streamlit application, named "QuLab," provides a hands-on, interactive environment to delve into the fascinating world of **phonetics** and its crucial role in **Text-to-Speech (TTS)** technology. TTS systems, which convert written text into audible speech, rely heavily on understanding fundamental speech units called **phonemes** and their measurable acoustic characteristics like **duration ($\text{ms}$)**, **pitch ($\text{Hz}$)**, and **energy**. These features are vital for producing natural-sounding and expressive speech (prosody).

The lab project aims to bridge the gap between theoretical phonetic concepts and their practical implications in speech technology and language education. It uses a synthetic dataset to simulate real-world phonetic phenomena, allowing users to:

*   **Generate and explore synthetic phonetic datasets** resembling phoneme characteristics.
*   **Visualize the distributions and relationships** between various phoneme features.
*   **Simulate latent feature creation and apply clustering** to identify natural phoneme groupings.
*   **Build a simplified predictive model** for "pronunciation naturalness" based on acoustic features.
*   **Interact with individual phoneme data** to compare characteristics against overall trends.

Ultimately, this application serves as a foundational tool to understand how intricate phonetic features collectively contribute to the naturalness and intelligibility of speech, paving the way for innovations in synthetic voice generation and interactive language learning tools.

## Features

The application is structured into three main navigation pages, each offering distinct functionalities:

### 1. Data Generation & Validation
*   **Synthetic Dataset Generation**: Create a flexible synthetic dataset of phoneme characteristics (duration, pitch, energy, naturalness score, context, dialect) with adjustable parameters like sample size and phoneme symbols.
*   **Data Overview**: Display the head of the generated DataFrame and its general information (`.info()`).
*   **Summary Statistics**: Present descriptive statistics (`.describe()`) for numeric columns.
*   **Data Validation**: Perform checks for expected column names, data types, and missing values in critical fields to ensure data quality and reliability.

### 2. Visualizing Relationships & Comparisons
*   **Distribution Histograms**: Visualize the individual distributions of key phoneme characteristics (`duration_ms`, `avg_pitch_hz`, `max_energy`) using interactive Plotly histograms, with adjustable bin count.
*   **Relationship Scatter Plots**: Explore pairwise relationships between features (e.g., duration vs. pitch, energy vs. pitch), with optional coloring by categorical variables (`is_vowel`, `dialect`) to reveal underlying patterns.
*   **Interactive Pair Plot**: Generate a comprehensive `plotly.express.scatter_matrix` for multiple selected numeric features, allowing for simultaneous visualization of pairwise relationships and distributions, with optional categorical coloring.
*   **Categorical Bar Comparisons**: Display bar charts to compare average phoneme characteristics (duration, pitch, naturalness score) across different categories like `phoneme_symbol`, `is_vowel`, and `dialect`.

### 3. Advanced Analysis & Interactive Tools
*   **Concept of Latent Features**: Introduction to latent features/embeddings in TTS, explaining their role in dimensionality reduction and complex relationship modeling.
*   **Simulating Feature Embeddings**: Generate simplified "latent features" by standardizing raw numeric phoneme characteristics using `StandardScaler`.
*   **K-Means Clustering**: Apply K-Means clustering on the simulated latent features to identify natural groupings of phonemes based on their acoustic properties. Users can adjust the number of clusters (`k`).
*   **Synthetic Naturalness Score**: Introduction to `pronunciation_naturalness_score` as a synthetic target variable for speech quality assessment.
*   **Modeling Phoneme Naturalness**: Train and evaluate a simple Linear Regression model to predict the `pronunciation_naturalness_score` from acoustic features, demonstrating how features contribute to perceived speech quality. Model coefficients, R-squared, and Mean Absolute Error (MAE) are displayed.
*   **Interactive Phoneme Analysis**: An interactive tool allowing users to select a specific phoneme symbol and instantly view its average characteristics (duration, pitch, energy, naturalness score) and compare them against overall dataset averages through a dynamic bar chart.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `git` (for cloning the repository)

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/phoneme-characteristics-analysis.git
    cd phoneme-characteristics-analysis
    ```
    *(Note: Replace `yourusername/phoneme-characteristics-analysis.git` with the actual repository URL if different.)*

2.  **Create a Virtual Environment**: (Recommended)
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**:
    Create a `requirements.txt` file in the root directory of the project with the following content:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    matplotlib
    seaborn
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit Application**:
    Ensure your virtual environment is activated and you are in the project's root directory (where `app.py` is located).
    ```bash
    streamlit run app.py
    ```

2.  **Access the Application**:
    The application will open in your default web browser (or provide a URL to copy-paste), usually at `http://localhost:8501`.

3.  **Navigate and Interact**:
    *   Use the sidebar **"Navigation"** dropdown menu to switch between the different sections of the application:
        *   `Data Generation & Validation`
        *   `Visualizing Relationships & Comparisons`
        *   `Advanced Analysis & Interactive Tools`
    *   On the `Data Generation & Validation` page, adjust parameters in the sidebar and click **"Generate Synthetic Data"** to begin.
    *   Explore the various sections, interact with sliders, dropdowns, and buttons to generate plots, perform analyses, and gain insights into phoneme characteristics.

## Project Structure

```
.
├── app.py                      # Main Streamlit application entry point
├── application_pages/
│   ├── __init__.py             # Initializes the application_pages directory as a Python package
│   ├── page1.py                # Contains Streamlit code for Data Generation & Validation
│   ├── page2.py                # Contains Streamlit code for Visualizing Relationships & Comparisons
│   └── page3.py                # Contains Streamlit code for Advanced Analysis & Interactive Tools
├── requirements.txt            # Lists Python dependencies
├── README.md                   # This file
└── .gitignore                  # Specifies intentionally untracked files to ignore
```

## Technology Stack

*   **Web Framework**: [Streamlit](https://streamlit.io/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Numerical Operations**: [NumPy](https://numpy.org/)
*   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (for `StandardScaler`, `KMeans`, `LinearRegression`, `train_test_split`, `r2_score`, `mean_absolute_error`)
*   **Data Visualization**:
    *   [Plotly Express](https://plotly.com/python/plotly-express/) & [Plotly Graph Objects](https://plotly.com/python/graph-objects/) (for interactive plots on Page 2 and Page 3)
    *   [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) (for styling and plots on Page 1, specifically for `df.info()` context and theme setup, though primary plots in other pages use Plotly)
*   **Utility**: `io` (for capturing stream output)

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository.
2.  **Clone** your forked repository.
3.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/fix-bug-name`.
4.  **Make your changes** and ensure they adhere to the project's coding style.
5.  **Commit your changes** with a clear and descriptive message: `git commit -m "feat: Add new feature X"` or `fix: Resolve bug Y"`.
6.  **Push to your branch**: `git push origin feature/your-feature-name`.
7.  **Open a Pull Request** against the `main` branch of the original repository.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
*(Note: A `LICENSE` file should be created in the root directory if not already present.)*

## Contact

For any questions or inquiries, please reach out via:
*   **Email**: [your.email@example.com](mailto:your.email@example.com)
*   **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

---