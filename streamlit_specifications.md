
# Streamlit Application Requirements Specification: Phoneme Characteristics Analyzer

This document outlines the requirements for developing a Streamlit application based on the provided Jupyter Notebook content and user requirements. The application aims to provide an interactive exploration of phoneme characteristics for Text-to-Speech (TTS) understanding and language learning contexts.

---

## 1. Application Overview

The Streamlit application will serve as an interactive tool for understanding phonetics and its application in Text-to-Speech (TTS) technology. It will allow users to generate synthetic phoneme data, visualize its characteristics, perform basic machine learning analyses, and interactively explore individual phoneme properties.

### Business Value
This application delves into the foundational elements of speech—phonemes—and their acoustic characteristics (duration, pitch, and energy). Understanding these characteristics is crucial for developing advanced Text-to-Speech (TTS) systems that generate natural-sounding speech and for creating effective language learning tools, such as an interactive phoneme trainer. By analyzing these features, the application helps identify patterns, distinguish between sounds, and ultimately enhance the quality and naturalness of synthetic speech.

### Learning Goals
Through this application, users will aim to:
*   **Understand fundamental concepts** related to phonetics, TTS components, and key speech characteristics.
*   **Gain insights** into how phoneme-level features like duration, pitch, and energy contribute to speech properties.
*   **Learn to generate, validate, and explore** synthetic datasets resembling phoneme characteristics.
*   **Practice data visualization techniques** to identify trends, relationships, and categorical comparisons in phoneme data.
*   **Explore simplified analytical models** (e.g., clustering, regression) to categorize phonemes or predict synthetic speech quality.
*   **Develop an understanding** of how interactive elements can aid in data exploration for language learning contexts.

---

## 2. User Interface Requirements

The application will feature a clear, intuitive layout, allowing users to navigate through different analytical steps and interact with the data and models.

### Layout and Navigation Structure
*   **Main Title and Introduction**: A prominent title and introductory text (including Business Value and Learning Goals) at the top of the main content area.
*   **Sidebar for Global Controls**: A dedicated sidebar (`st.sidebar`) for global application settings, dataset generation parameters, and potentially navigation between major sections if the application becomes very long.
*   **Section-based Content Organization**: The main content area will be organized into distinct, logically flowing sections, corresponding to the notebook's structure. Each major section will have a clear header (`st.header` or `st.subheader`) and narrative description (`st.markdown`).
*   **Expanders for Details**: Use `st.expander` components to collapse/expand detailed descriptions, code stubs, and potentially less critical output, allowing users to focus on primary results.
*   **Display Components**:
    *   Markdown for narrative text, descriptions, and mathematical equations.
    *   `st.dataframe` for displaying tabular data (e.g., `df.head()`, `df.describe()`).
    *   `st.pyplot` for displaying Matplotlib/Seaborn plots.
    *   `st.write` or `st.info`/`st.success`/`st.error` for validation messages and model evaluation results.

### Input Widgets and Controls
The following input widgets will be used to enable user interaction and parameter adjustment:

1.  **Synthetic Dataset Generation Parameters (Sidebar)**:
    *   `num_samples`: `st.number_input` (e.g., range 0-5000, default 1000).
        *   *Help Text*: "Number of synthetic phoneme samples to generate."
    *   `phoneme_symbols`: `st.multiselect` with predefined options.
        *   *Help Text*: "List of phoneme symbols to include in the dataset."
    *   `word_contexts`: `st.multiselect` with predefined options.
        *   *Help Text*: "List of word contexts for phoneme instances."
    *   `dialects`: `st.multiselect` with predefined options.
        *   *Help Text*: "List of dialects to simulate for pronunciation variations."
    *   `random_seed`: `st.number_input` (e.g., default 42).
        *   *Help Text*: "Random seed for reproducible data generation."
    *   **Action Button**: `st.button("Generate Synthetic Data")`.
        *   *Help Text*: "Click to generate the phoneme dataset with the specified parameters."

2.  **Visualization Parameters (within respective sections)**:
    *   **Histogram Bins**: `st.slider` for `bins` (e.g., range 10-50, default 30).
        *   *Help Text*: "Number of bins for the histogram plots."
    *   **Pair Plot Features**: `st.multiselect` to select numeric features to include.
        *   *Help Text*: "Select numeric features for the comprehensive pair plot visualization."
    *   **Pair Plot Hue**: `st.selectbox` for coloring points by a categorical variable.
        *   *Help Text*: "Select a categorical column to color the points in the pair plot."

3.  **Clustering Parameters**:
    *   `NUM_CLUSTERS`: `st.slider` or `st.number_input` (e.g., range 2-10, default 4).
        *   *Help Text*: "Number of clusters ($k$) for the K-Means algorithm."
    *   **Action Button**: `st.button("Perform Clustering")`.
        *   *Help Text*: "Click to apply K-Means clustering to the latent features."

4.  **Modeling Parameters**:
    *   **Action Button**: `st.button("Train and Evaluate Regression Model")`.
        *   *Help Text*: "Click to train a linear regression model and evaluate its performance."

5.  **Interactive Phoneme Analyzer**:
    *   `phoneme_symbol_selector`: `st.selectbox` populated with unique phoneme symbols from the generated data.
        *   *Help Text*: "Select a phoneme symbol to view its average characteristics and compare them against the overall dataset averages."

### Visualization Components (charts, graphs, tables)
All plots will adhere to the "Style & Usability" requirements from the user specification (color-blind-friendly palette, font size $\geq$ 12 pt, clear titles, labeled axes, legends).

1.  **DataFrame Information & Statistics**:
    *   `phoneme_df.head()`: Displayed using `st.dataframe`.
    *   `phoneme_df.info()`: Displayed as preformatted text (`st.text`).
    *   `phoneme_df.describe()`: Displayed using `st.dataframe`.
2.  **Histograms (Distribution Plots)**:
    *   Plots for `duration_ms`, `avg_pitch_hz`, `max_energy`.
    *   Displayed using `st.pyplot`.
3.  **Scatter Plots (Relationship Plots)**:
    *   `duration_ms` vs. `avg_pitch_hz`, colored by `is_vowel`.
    *   `max_energy` vs. `avg_pitch_hz`, colored by `dialect`.
    *   Displayed using `st.pyplot`.
4.  **Pair Plot**:
    *   `duration_ms`, `avg_pitch_hz`, `max_energy`, `pronunciation_naturalness_score`, with `is_vowel` as hue.
    *   Displayed using `st.pyplot`.
5.  **Bar Charts (Categorical Comparisons)**:
    *   Average `duration_ms` per `phoneme_symbol`.
    *   Average `avg_pitch_hz` per `dialect`.
    *   Average `pronunciation_naturalness_score` for `is_vowel` (True/False).
    *   Displayed using `st.pyplot`.
6.  **Clustering Visualization**:
    *   Scatter plot of `duration_ms` vs. `avg_pitch_hz`, colored by `cluster_label`.
    *   Displayed using `st.pyplot`.
7.  **Interactive Phoneme Analyzer Output**:
    *   Text display of average characteristics for the selected phoneme.
    *   Bar chart comparing selected phoneme's characteristics against overall dataset averages.
    *   Displayed using `st.markdown` and `st.pyplot`.

### Interactive Elements and Feedback Mechanisms
*   **Dynamic Plot Updates**: All plots should automatically re-render when their associated input widgets are changed.
*   **Real-time Validation Feedback**: Display `st.success` or `st.error` messages for data validation.
*   **Model Performance Display**: R-squared and MAE scores displayed after model training and evaluation.
*   **Loading Indicators**: Use `st.spinner` or `st.status` for potentially long-running operations like data generation or pair plot creation.

---

## 3. Additional Requirements

### Annotation and Tooltip Specifications
*   **Input Controls**: All sliders, number inputs, multiselects, and dropdowns will include clear `help` arguments or descriptive labels explaining their purpose, as specified in "Input Widgets and Controls" above.
*   **Plot Annotations**: All plots will have descriptive titles, clearly labeled axes, and legends as per the notebook's plotting functions.
*   **Narrative Context**: `st.markdown` will be extensively used to provide the explanatory text from the Jupyter Notebook, guiding the user through each step of the analysis and explaining the "what" and "why."

### Save the States of the Fields Properly So That Changes Are Not Lost
*   **Session State**: The `st.session_state` API will be extensively used to maintain the state of:
    *   **Generated DataFrame**: `phoneme_df`.
    *   **Transformed DataFrames**: `latent_features_df`, `X_train`, `X_test`, `y_train`, `y_test`.
    *   **Trained Models**: `regression_model`.
    *   **Clustering Results**: `cluster_labels`.
    *   **User Input Parameters**: All values from input widgets (`num_samples`, `random_seed`, selected columns, cluster count, etc.).
    *   **Validation Status**: `data_validated`.
*   This ensures that when a user interacts with a widget, the application's state is preserved across reruns, preventing data loss and providing a seamless interactive experience.

---

## 4. Notebook Content and Code Requirements

This section details how the Jupyter Notebook's markdown and code will be integrated into the Streamlit application.

### General Setup and Configuration
The Streamlit application will start with necessary library imports and global configuration for plotting aesthetics.

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import io # For capturing df.info() output

# Configure Seaborn for better aesthetics and colorblind-friendly palette
sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
```

### Application Structure Mapping (Section by Section)

#### 4.1. Introduction to Phonetics and Text-to-Speech (TTS)
This section provides the overall context, business value, and learning goals of the application.

*   **Markdown Content**:
    ```python
    st.title("Phoneme Characteristics Analysis: An Interactive Exploration for Text-to-Speech Understanding")
    st.markdown("""
    This application delves into the fascinating world of **phonetics**, the scientific study of speech sounds, and its crucial role in **Text-to-Speech (TTS)** technology. TTS systems, which convert written text into audible speech, are becoming increasingly sophisticated, powering virtual assistants, accessibility tools, and various interactive applications.

    At the heart of speech are **phonemes**: "The smallest unit of sound that makes a word's pronunciation and meaning different from another word." These fundamental sound units are not just abstract concepts; they possess measurable characteristics like **duration**, **pitch**, and **energy**, which are vital for producing natural-sounding speech. These features contribute significantly to **prosody**, the rhythm, stress, and intonation of speech, making the generated voice expressive and intelligible.

    In a typical TTS system, as illustrated in many research papers (e.g., Fig. 1 in "A review-based study on different Text-to-Speech technologies"), components like the preprocessor, encoder, decoder, and vocoder work in concert. The encoder, for instance, often takes linguistic features, including phonemes and their characteristics, and transforms them into more abstract "latent features" that the decoder then uses to generate mel-spectrograms (visual representations of sound frequencies over time), ultimately leading to the synthesized speech.

    ### Business Value:
    This application delves into the foundational elements of speech—phonemes—and their acoustic characteristics (duration, pitch, and energy). Understanding these characteristics is crucial for developing advanced Text-to-Speech (TTS) systems that generate natural-sounding speech and for creating effective language learning tools, such as an interactive phoneme trainer. By analyzing these features, we can identify patterns, distinguish between sounds, and ultimately enhance the quality and naturalness of synthetic speech.

    ### What We Will Be Covering / Learning:
    In this application, we will explore:
    *   **Fundamental concepts** in phonetics and Text-to-Speech (TTS) technology.
    *   How **phoneme-level features** (duration, pitch, energy) influence speech properties.
    *   Methods to **generate, validate, and explore synthetic datasets** resembling phoneme characteristics.
    *   **Data visualization techniques** to uncover trends and relationships in phoneme data.
    *   **Simplified analytical models** (clustering, regression) to categorize phonemes and predict synthetic speech quality.
    *   The implementation of **interactive elements** to facilitate data exploration, particularly useful for language learning contexts.

    Our goal is to simulate and understand the intricate relationship between phonetic features and speech perception, paving the way for more sophisticated TTS applications and educational tools.
    """)
    ```

#### 4.2. Synthetic Dataset Generation: Phoneme Characteristics
This section allows users to generate a synthetic dataset representing phoneme characteristics using adjustable parameters.

*   **Markdown Content**:
    ```python
    st.header("1. Synthetic Dataset Generation: Phoneme Characteristics")
    st.markdown("""
    Due to the complexities and resource intensiveness of acquiring and processing real-world phonetic data, especially for a focused study, we will generate a **synthetic dataset**. This synthetic data allows us to simulate realistic phoneme characteristics and their interrelationships, providing a controlled environment for learning and analysis without the overhead of complex audio processing pipelines.

    Our synthetic dataset will include the following key features, inspired by linguistic features discussed in Text-to-Speech research:

    *   `phoneme_id`: A unique identifier for each phoneme instance.
    *   `phoneme_symbol`: The linguistic symbol representing the phoneme (e.g., 'a', 'b', 'sh').
    *   `word_context`: The word in which the phoneme appears (to simulate contextual variations).
    *   `duration_ms`: The duration of the phoneme in milliseconds ($\text{ms}$). Phoneme duration is a crucial aspect of speech rhythm and naturalness.
    *   `avg_pitch_hz`: The average pitch (fundamental frequency) of the phoneme in Hertz ($\text{Hz}$). Pitch is key to conveying emotions and affects speech prosody.
    *   `max_energy`: The maximum energy of the phoneme (arbitrary units). Energy relates to the volume and intensity of the sound, impacting prosody.
    *   `is_vowel`: A boolean indicating whether the phoneme is a vowel or a consonant, as vowels and consonants often have distinct acoustic properties.
    *   `dialect`: The simulated dialect, influencing subtle variations in pronunciation.
    *   `pronunciation_naturalness_score`: A synthetic target variable, ranging from 0 to 100, representing how 'natural' or 'well-formed' the phoneme's pronunciation is. This score is a proxy for perceived speech quality, relevant for applications like a phoneme trainer.

    The `generate_synthetic_phoneme_data` function below creates this dataset, ensuring variability and introducing realistic correlations between these features, such as vowels generally having longer durations and higher energy than consonants.
    """)
    ```
*   **Code Stub (`generate_synthetic_phoneme_data` function and Streamlit UI)**:
    ```python
    @st.cache_data # Cache the generated data to prevent re-running on every interaction
    def generate_synthetic_phoneme_data(num_samples, phoneme_symbols, word_contexts, dialects, random_seed):
        """
        Generates a Pandas DataFrame with synthetic phoneme data, including features
        like duration, pitch, energy, and a naturalness score, with variations
        based on phoneme type (vowel/consonant) and specific phoneme symbols.
        """
        # --- Input Validation ---
        if not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer.")
        if num_samples < 0:
            raise ValueError("num_samples cannot be negative.")
        
        if not isinstance(phoneme_symbols, list) or not all(isinstance(s, str) for s in phoneme_symbols):
            raise TypeError("phoneme_symbols must be a list of strings.")
        if not isinstance(word_contexts, list) or not all(isinstance(s, str) for s in word_contexts):
            raise TypeError("word_contexts must be a list of strings.")
        if not isinstance(dialects, list) or not all(isinstance(s, str) for s in dialects):
            raise TypeError("dialects must be a list of strings.")
        
        if not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer.")

        if num_samples > 0:
            if not phoneme_symbols:
                raise ValueError("phoneme_symbols cannot be empty if num_samples > 0.")
            if not word_contexts:
                raise ValueError("word_contexts cannot be empty if num_samples > 0.")
            if not dialects:
                raise ValueError("dialects cannot be empty if num_samples > 0.")

        # Define expected column names for consistency, especially for num_samples = 0
        expected_column_names = [
            'phoneme_id', 'phoneme_symbol', 'word_context', 'duration_ms', 'avg_pitch_hz', 
            'max_energy', 'is_vowel', 'dialect', 'pronunciation_naturalness_score'
        ]

        # Handle num_samples = 0 by returning an empty DataFrame with the correct columns
        if num_samples == 0:
            return pd.DataFrame(columns=expected_column_names)

        np.random.seed(random_seed)

        # Define common vowels for `is_vowel` determination
        VOWELS = {'a', 'e', 'i', 'o', 'u'}

        # Data structure to collect generated samples
        data_records = []

        # Base feature parameters (mean, standard deviation) for vowels vs. consonants
        base_feature_params = {
            'vowel': {
                'duration_ms': (120, 30),
                'avg_pitch_hz': (160, 45),
                'max_energy': (0.85, 0.15)
            },
            'consonant': {
                'duration_ms': (60, 20),
                'avg_pitch_hz': (90, 35),
                'max_energy': (0.45, 0.1)
            }
        }

        # Phoneme-specific micro-adjustments
        phoneme_specific_adjustments = {
            'a': {'duration_ms': 15, 'avg_pitch_hz': 10, 'max_energy': 0.05},
            'e': {'duration_ms': -5, 'avg_pitch_hz': 5, 'max_energy': -0.02},
            'i': {'duration_ms': 0, 'avg_pitch_hz': 12, 'max_energy': 0.03},
            'o': {'duration_ms': 10, 'avg_pitch_hz': -5, 'max_energy': 0.01},
            'u': {'duration_ms': 5, 'avg_pitch_hz': -8, 'max_energy': 0.04},
            'p': {'duration_ms': 10, 'avg_pitch_hz': -15, 'max_energy': 0.08},
            't': {'duration_ms': -5, 'avg_pitch_hz': -10, 'max_energy': 0.03},
            'k': {'duration_ms': 5, 'avg_pitch_hz': -12, 'max_energy': 0.06},
        }

        # Coefficients for the linear combination of features for naturalness score
        naturalness_score_coeffs = {
            'duration_ms': 0.15, 
            'avg_pitch_hz': 0.08, 
            'max_energy': 60     
        }
        naturalness_score_base = 25 # A baseline score
        naturalness_score_noise_std = 7 # Noise for the score

        for i in range(num_samples):
            record = {}
            record['phoneme_id'] = i + 1
            record['phoneme_symbol'] = np.random.choice(phoneme_symbols)
            record['word_context'] = np.random.choice(word_contexts)
            record['dialect'] = np.random.choice(dialects)
            
            record['is_vowel'] = record['phoneme_symbol'].lower() in VOWELS
            
            category_key = 'vowel' if record['is_vowel'] else 'consonant'
            params = base_feature_params[category_key]

            current_duration_mean, current_duration_std = params['duration_ms']
            current_pitch_mean, current_pitch_std = params['avg_pitch_hz']
            current_energy_mean, current_energy_std = params['max_energy']

            if record['phoneme_symbol'] in phoneme_specific_adjustments:
                adj = phoneme_specific_adjustments[record['phoneme_symbol']]
                current_duration_mean += adj.get('duration_ms', 0)
                current_pitch_mean += adj.get('avg_pitch_hz', 0)
                current_energy_mean += adj.get('max_energy', 0)

            record['duration_ms'] = max(0, np.random.normal(current_duration_mean, current_duration_std))
            record['avg_pitch_hz'] = max(0, np.random.normal(current_pitch_mean, current_pitch_std))
            record['max_energy'] = max(0, np.random.normal(current_energy_mean, current_energy_std))

            score = (
                naturalness_score_base +
                record['duration_ms'] * naturalness_score_coeffs['duration_ms'] +
                record['avg_pitch_hz'] * naturalness_score_coeffs['avg_pitch_hz'] +
                record['max_energy'] * naturalness_score_coeffs['max_energy'] +
                np.random.normal(0, naturalness_score_noise_std)
            )
            record['pronunciation_naturalness_score'] = np.clip(score, 0, 100)

            data_records.append(record)

        df = pd.DataFrame(data_records)
        return df[expected_column_names]

    # Input Widgets in Sidebar
    with st.sidebar:
        st.subheader("Data Generation Parameters")
        num_samples = st.number_input(
            "Number of Samples", min_value=0, max_value=5000, value=st.session_state.get('num_samples', 1000), step=100, 
            help="Number of synthetic phoneme samples to generate."
        )
        phoneme_symbols_default = ['a', 'e', 'i', 'o', 'u', 'b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'z']
        phoneme_symbols_input = st.multiselect(
            "Phoneme Symbols", options=phoneme_symbols_default, default=st.session_state.get('phoneme_symbols_input', phoneme_symbols_default),
            help="Select phoneme symbols to include in the dataset."
        )
        word_contexts_default = ['cat', 'dog', 'house', 'run', 'speak', 'phone', 'data', 'learn']
        word_contexts_input = st.multiselect(
            "Word Contexts", options=word_contexts_default, default=st.session_state.get('word_contexts_input', word_contexts_default),
            help="Select word contexts for phoneme instances."
        )
        dialects_default = ['General American', 'British English', 'Australian English']
        dialects_input = st.multiselect(
            "Dialects", options=dialects_default, default=st.session_state.get('dialects_input', dialects_default),
            help="Select dialects to simulate for pronunciation variations."
        )
        random_seed = st.number_input(
            "Random Seed", min_value=0, value=st.session_state.get('random_seed', 42), step=1,
            help="Random seed for reproducible data generation."
        )

        if st.button("Generate Synthetic Data", key="generate_data_button"):
            with st.spinner("Generating data..."):
                st.session_state['phoneme_df'] = generate_synthetic_phoneme_data(
                    num_samples, phoneme_symbols_input, word_contexts_input, dialects_input, random_seed
                )
                st.session_state['num_samples'] = num_samples
                st.session_state['phoneme_symbols_input'] = phoneme_symbols_input
                st.session_state['word_contexts_input'] = word_contexts_input
                st.session_state['dialects_input'] = dialects_input
                st.session_state['random_seed'] = random_seed
            st.success("Synthetic phoneme data generated!")
    
    # Display the generated DataFrame head
    if 'phoneme_df' in st.session_state:
        st.subheader("First 5 rows of the synthetic phoneme data:")
        st.dataframe(st.session_state['phoneme_df'].head())
        st.markdown(f"""
        The output above confirms the successful generation of our synthetic dataset with {len(st.session_state['phoneme_df'])} samples. We can observe the various columns as described:

        *   `phoneme_id`: Unique identifier for each entry.
        *   `phoneme_symbol`: The specific phoneme, e.g., 's', 'a', 'e'.
        *   `word_context`: The simulated word context.
        *   `duration_ms`: Phoneme duration in milliseconds, varying as expected.
        *   `avg_pitch_hz`: Average pitch in Hertz.
        *   `max_energy`: Maximum energy level.
        *   `is_vowel`: A boolean indicating if the phoneme is a vowel, which is crucial for distinguishing characteristics.
        *   `dialect`: The assigned dialect.
        *   `pronunciation_naturalness_score`: The derived naturalness score, a key metric for our analysis.

        This dataset now provides a foundation for exploring phonetic characteristics and their implications for speech naturalness and TTS systems.
        """)
    else:
        st.info("Adjust parameters in the sidebar and click 'Generate Synthetic Data' to begin.")
    ```

#### 4.3. Exploring the Synthetic Phoneme Data
This section provides an initial overview of the generated dataset's structure and basic statistics.

*   **Markdown Content**:
    ```python
    st.header("2. Exploring the Synthetic Phoneme Data")
    st.markdown("""
    Initial data exploration is a crucial step to understand the structure, content, and basic statistics of our dataset. This helps us ensure that the synthetic data generation process has yielded results that align with our expectations for further analysis related to phoneme characteristics. We'll examine data types, non-null counts, and summary statistics for numeric columns.
    """)
    ```
*   **Code Stub**:
    ```python
    if 'phoneme_df' in st.session_state:
        st.subheader("DataFrame Information:")
        # Capture phoneme_df.info() output as a string
        buffer = io.StringIO()
        st.session_state['phoneme_df'].info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Summary Statistics for Numeric Columns:")
        st.dataframe(st.session_state['phoneme_df'].describe())
        st.markdown("""
        The `phoneme_df.info()` output confirms that our DataFrame contains `num_samples` entries with no missing values, as indicated by the 'Non-Null Count' matching the total number of entries for all columns. The data types (`int64`, `object`, `float64`, `bool`) are as expected for each feature.

        From `phoneme_df.describe()`, we can observe:

        *   **`duration_ms`**: Ranges from approximately 13ms to 186ms, with an average around 90ms. This shows a reasonable spread for phoneme durations, distinguishing between shorter consonants and longer vowels.
        *   **`avg_pitch_hz`**: Ranges from 15Hz to 273Hz, averaging around 125Hz. This covers a plausible range for human speech pitch.
        *   **`max_energy`**: Ranges from about 0.08 to 1.13, with an average around 0.6. This represents a good variation in phoneme intensity.
        *   **`pronunciation_naturalness_score`**: Ranges from 11 to 99, with an average of approximately 66. This indicates that our synthetic naturalness scores are spread across a meaningful range, reflecting varying degrees of 'naturalness'.

        These statistics suggest that our synthetic data successfully captures a realistic range and variability of phoneme characteristics, making it suitable for further analysis and modeling experiments.
        """)
    else:
        st.info("Generate synthetic data first to explore it.")
    ```

#### 4.4. Data Validation and Summary Statistics
This section performs critical checks on the dataset's integrity and quality.

*   **Markdown Content**:
    ```python
    st.header("3. Data Validation and Summary Statistics")
    st.markdown("""
    Data validation is a critical step to ensure the quality and reliability of our dataset before proceeding with any in-depth analysis or modeling. By systematically checking for expected column names, verifying data types, and asserting the absence of missing values in `critical_fields`, we can prevent downstream errors and ensure the integrity of our phoneme characteristic data.

    For this dataset, we consider `duration_ms`, `avg_pitch_hz`, and `max_energy` as `critical_fields` because these are fundamental acoustic properties that must be present and valid for any meaningful phonetic analysis.
    """)
    ```
*   **Code Stub (`validate_and_summarize_data` function and call)**:
    ```python
    def validate_and_summarize_data(df, expected_columns, expected_dtypes, critical_fields):
        """
        Performs data validation and logs summary statistics.
        """
        st.subheader("--- Starting Data Validation ---")
        validation_passed = True
        
        try:
            # 1. Validate Expected Column Names
            actual_columns_set = set(df.columns)
            expected_columns_set = set(expected_columns)
            
            missing_from_df = expected_columns_set - actual_columns_set
            if missing_from_df:
                st.error(f"Validation Error: Expected columns missing from DataFrame: {sorted(list(missing_from_df))}. "
                         f"Actual columns: {sorted(list(df.columns))}.")
                validation_passed = False
            else:
                st.success(f"Validation Step 1: All {len(expected_columns)} expected columns are present.")

            # 2. Validate Data Types
            for col, expected_dtype in expected_dtypes.items():
                if col not in df.columns: # Should be covered by first check, but defensive
                    st.error(f"Validation Error: Column '{col}' in expected_dtypes not found in DataFrame.")
                    validation_passed = False
                    continue
                actual_dtype = df[col].dtype
                if not pd.api.types.is_dtype_equal(actual_dtype, expected_dtype):
                    st.error(f"Validation Error: Data type mismatch for column '{col}'. "
                             f"Expected '{expected_dtype}', got '{actual_dtype}'.")
                    validation_passed = False
            if validation_passed: # Only show success if all type checks passed so far
                st.success("Validation Step 2: All column data types match expected types.")

            # 3. Validate Missing Values in Critical Fields
            for field in critical_fields:
                if field not in df.columns: # Should be covered by first check, but defensive
                    st.error(f"Validation Error: Critical field '{field}' not found in DataFrame.")
                    validation_passed = False
                    continue
                if df[field].isnull().any():
                    missing_indices = df.index[df[field].isnull()].tolist()
                    st.error(f"Validation Error: Missing values found in critical field '{field}' "
                             f"at index/indices: {missing_indices}.")
                    validation_passed = False
            if validation_passed: # Only show success if all missing value checks passed so far
                st.success("Validation Step 3: No missing values in critical fields.")

            if validation_passed:
                st.success("--- Data Validation Complete: All checks passed ---")
            else:
                st.error("--- Data Validation Complete: Some checks failed ---")

        except Exception as e:
            st.error(f"An unexpected error occurred during validation: {e}")
            validation_passed = False

        # Summarize Data regardless of validation pass/fail
        st.subheader("--- Data Summary ---")
        if df.empty:
            st.info("DataFrame is empty. No summary statistics to display.")
        else:
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                st.write("Summary statistics for numeric columns:")
                st.dataframe(numeric_df.describe())
            else:
                st.info("No numeric columns found in the DataFrame to summarize.")
        st.write("--- Summary Complete ---")
        return validation_passed

    if 'phoneme_df' in st.session_state:
        expected_columns = [
            'phoneme_id', 'phoneme_symbol', 'word_context', 'duration_ms',
            'avg_pitch_hz', 'max_energy', 'is_vowel', 'dialect', 'pronunciation_naturalness_score'
        ]
        expected_dtypes = {
            'phoneme_id': 'int64', 'phoneme_symbol': 'object', 'word_context': 'object',
            'duration_ms': 'float64', 'avg_pitch_hz': 'float64', 'max_energy': 'float64',
            'is_vowel': 'bool', 'dialect': 'object', 'pronunciation_naturalness_score': 'float64'
        }
        critical_fields = ['duration_ms', 'avg_pitch_hz', 'max_energy']
        
        st.session_state['data_validated'] = validate_and_summarize_data(
            st.session_state['phoneme_df'], expected_columns, expected_dtypes, critical_fields
        )
        st.markdown("""
        The validation results confirm that our synthetic `phoneme_df` meets the defined quality standards:

        *   **All expected columns are present**: This ensures that our DataFrame has the complete set of features required for analysis.
        *   **All column data types match expected types**: Correct data types are crucial for proper numerical operations and categorical groupings.
        *   **No missing values in critical fields**: The absence of `NaN`s in `duration_ms`, `avg_pitch_hz`, and `max_energy` guarantees that our core acoustic features are complete and reliable.

        The summary statistics for numeric columns, already discussed in the previous section, are reiterated here, reinforcing our understanding of the dataset's central tendencies, spread, and overall realism for synthetic data.
        """)
    else:
        st.info("Generate synthetic data first to validate it.")
    ```

#### 4.5. Visualizing Phoneme Durations and Frequencies
This section provides histograms to understand the distribution of key phoneme characteristics.

*   **Markdown Content**:
    ```python
    st.header("4. Visualizing Phoneme Durations and Frequencies")
    st.markdown("""
    Visualizing the distribution of phoneme characteristics is a fundamental step in phonetic analysis. It helps us understand the typical ranges, variability, and overall patterns within features like duration, pitch, and energy. For instance, `duration_ms` is a key feature influencing the rhythm and clarity of speech, as noted in discussions of "Phoneme duration" in TTS research.

    We will generate histograms for `duration_ms`, `avg_pitch_hz`, and `max_energy` to observe their individual distributions.
    """)
    ```
*   **Code Stub (`plot_distribution_histogram` function and calls)**:
    ```python
    def plot_distribution_histogram(df, column_name, title, x_label, y_label, bins=30):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x=column_name, bins=bins, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig

    if st.session_state.get('data_validated', False):
        st.subheader("Distribution Histograms")
        bins_input = st.slider("Number of Bins for Histograms", min_value=10, max_value=50, value=30, step=5,
                               help="Adjust the number of bins to see different levels of detail in the distributions.")

        col1, col2, col3 = st.columns(3)
        with col1:
            fig1 = plot_distribution_histogram(
                st.session_state['phoneme_df'], 'duration_ms', 'Distribution of Phoneme Duration (ms)', 'Duration (ms)', 'Frequency', bins=bins_input
            )
            st.pyplot(fig1)
            plt.close(fig1) # Close plot to free memory
        with col2:
            fig2 = plot_distribution_histogram(
                st.session_state['phoneme_df'], 'avg_pitch_hz', 'Distribution of Average Pitch (Hz)', 'Average Pitch (Hz)', 'Frequency', bins=bins_input
            )
            st.pyplot(fig2)
            plt.close(fig2)
        with col3:
            fig3 = plot_distribution_histogram(
                st.session_state['phoneme_df'], 'max_energy', 'Distribution of Maximum Energy', 'Maximum Energy', 'Frequency', bins=bins_input
            )
            st.pyplot(fig3)
            plt.close(fig3)
        st.markdown("""
        The generated histograms provide insights into the synthetic distributions of key phoneme characteristics:

        *   **Duration ($\text{ms}$)**: The histogram for `duration_ms` appears to be a bimodal or multimodal distribution, reflecting the synthetic distinction between generally shorter consonants and longer vowels. The peaks align with the mean durations set for vowels (around 120ms) and consonants (around 60ms).
        *   **Average Pitch ($\text{Hz}$)**: The `avg_pitch_hz` distribution also shows distinct patterns, likely influenced by the synthetic assignment of different pitch ranges to vowels and consonants, or specific phonemes. This aligns with the understanding that pitch varies significantly across different speech sounds.
        *   **Maximum Energy**: The distribution of `max_energy` also indicates clear differences between phoneme types, with vowels typically exhibiting higher energy. The histogram shows a spread that covers both lower-energy consonants and higher-energy vowels.

        These distributions confirm that our synthetic data successfully introduces realistic variability and categorical distinctions, which is crucial for simulating phonetic phenomena for a phoneme trainer or TTS system. The varied shapes imply that these features are not uniformly distributed and are influenced by underlying factors, such as the `is_vowel` attribute and specific `phoneme_symbol` adjustments.
        """)
    else:
        st.info("Perform data generation and validation first to visualize distributions.")
    ```

#### 4.6. Analyzing Relationships: Pitch, Energy, and Duration
This section uses scatter plots and a pair plot to visualize pairwise relationships and correlations between features.

*   **Markdown Content**:
    ```python
    st.header("5. Analyzing Relationships: Pitch, Energy, and Duration")
    st.markdown("""
    Understanding the interrelationships between different speech features is vital for comprehending how phonemes are formed and perceived. For example, research highlights that "Pitch: Key feature to convey emotions, it greatly affects the speech prosody" and "Energy: Indicates frame-level magnitude of mel-spectrograms... affects the volume and prosody of speech." Exploring these correlations helps us build predictive models, identify patterns in phoneme pronunciation, and understand the acoustic cues that differentiate sounds.

    We will use scatter plots to visualize pairwise relationships, optionally coloring points by categorical variables like `is_vowel` or `dialect` to observe how these categories influence the relationships. Additionally, a `seaborn.pairplot` will provide a comprehensive overview of all pairwise relationships and distributions for multiple numeric features.
    """)
    ```
*   **Code Stub (`plot_scatter_relationship`, `plot_features_pairplot` functions and calls)**:
    ```python
    def plot_scatter_relationship(df, x_column, y_column, hue_column, title, x_label, y_label):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, palette='colorblind', s=50, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.6)
        if hue_column:
            ax.legend(title=hue_column.replace('_', ' ').title())
        plt.tight_layout()
        return fig

    def plot_features_pairplot(df, features_list, hue_column=None, title=None):
        if df.empty and features_list:
            st.warning("Cannot plot an empty DataFrame with specified features.")
            return None
        
        g = sns.pairplot(df, vars=features_list, hue=hue_column, diag_kind='hist', palette='colorblind')
        if title:
            g.fig.suptitle(title, y=1.02, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        return g.fig

    if st.session_state.get('data_validated', False):
        st.subheader("Relationship Plots")

        st.write("#### Duration vs. Pitch, by Vowel/Consonant")
        fig_s1 = plot_scatter_relationship(
            st.session_state['phoneme_df'], 'duration_ms', 'avg_pitch_hz', 'is_vowel', 
            'Duration vs. Pitch, by Vowel/Consonant', 'Duration (ms)', 'Average Pitch (Hz)'
        )
        st.pyplot(fig_s1)
        plt.close(fig_s1)

        st.write("#### Max Energy vs. Pitch, by Dialect")
        fig_s2 = plot_scatter_relationship(
            st.session_state['phoneme_df'], 'max_energy', 'avg_pitch_hz', 'dialect', 
            'Max Energy vs. Pitch, by Dialect', 'Maximum Energy', 'Average Pitch (Hz)'
        )
        st.pyplot(fig_s2)
        plt.close(fig_s2)

        st.subheader("Pair Plot of Phoneme Characteristics")
        numeric_cols_for_pairplot = ['duration_ms', 'avg_pitch_hz', 'max_energy', 'pronunciation_naturalness_score']
        selected_pairplot_features = st.multiselect(
            "Select Features for Pair Plot", 
            options=numeric_cols_for_pairplot, 
            default=st.session_state.get('selected_pairplot_features', numeric_cols_for_pairplot),
            help="Choose the numeric features to display in the pair plot."
        )
        pairplot_hue_options = ['None'] + [col for col in st.session_state['phoneme_df'].select_dtypes(include='object').columns if col != 'phoneme_symbol'] + ['is_vowel']
        selected_pairplot_hue = st.selectbox(
            "Color Pair Plot by (Hue)", 
            options=pairplot_hue_options, 
            index=pairplot_hue_options.index(st.session_state.get('selected_pairplot_hue', 'is_vowel')),
            help="Select a categorical column to color the points in the pair plot."
        )
        pairplot_hue_val = None if selected_pairplot_hue == 'None' else selected_pairplot_hue

        if st.button("Generate Pair Plot", key="pairplot_button"):
            st.session_state['selected_pairplot_features'] = selected_pairplot_features
            st.session_state['selected_pairplot_hue'] = selected_pairplot_hue
            with st.spinner("Generating pair plot... This might take a moment."):
                fig_pair = plot_features_pairplot(
                    st.session_state['phoneme_df'],
                    selected_pairplot_features,
                    hue_column=pairplot_hue_val,
                    title='Pair Plot of Phoneme Characteristics'
                )
                if fig_pair:
                    st.pyplot(fig_pair)
                    plt.close(fig_pair)
                else:
                    st.warning("Could not generate pair plot. Check selected features/data.")

        st.markdown("""
        The scatter plots and the pair plot reveal interesting synthetic relationships within our phoneme data:

        *   **Duration vs. Pitch (by Vowel/Consonant)**: The scatter plot clearly shows a distinction between vowels and consonants. Vowels (True for `is_vowel`) tend to occupy the upper-right region of the plot, indicating generally longer durations and higher average pitches. Consonants (False for `is_vowel`) are typically in the lower-left, with shorter durations and lower pitches. This aligns with phonetic principles where vowels are often sustained longer and have a clearer fundamental frequency.

        *   **Max Energy vs. Pitch (by Dialect)**: This plot highlights how different synthetic `dialect` categories are distributed across energy and pitch ranges. While there's significant overlap, subtle clustering or shifts for certain dialects might be observed, reflecting simulated regional variations in speech production. For instance, one dialect might be characterized by slightly higher overall pitch or energy compared to another.

        *   **Pair Plot of Phoneme Characteristics**: The `pairplot` offers a holistic view:
            *   **Histograms on the diagonal** reaffirm the distributions seen earlier, often showing bimodal patterns for features like `duration_ms` and `avg_pitch_hz` when colored by `is_vowel`.
            *   **Scatter plots off-diagonal** further illustrate pairwise correlations. For example, there appears to be a positive correlation between `duration_ms` and `max_energy` (longer phonemes tend to be more energetic). The `pronunciation_naturalness_score` also shows positive correlations with these acoustic features, suggesting that 'naturalness' is synthetically linked to optimal ranges of duration, pitch, and energy.

        These visualizations are critical for understanding how different phonetic features interact and how categorical factors like vowel/consonant type or dialect influence these interactions. Such insights are fundamental for developing robust TTS systems or effective phoneme training tools.
        """)
    else:
        st.info("Perform data generation and validation first to analyze relationships.")
    ```

#### 4.7. Categorical Comparisons: Phoneme Type and Dialect
This section uses bar plots to compare average phoneme characteristics across different categorical groups.

*   **Markdown Content**:
    ```python
    st.header("6. Categorical Comparisons: Phoneme Type and Dialect")
    st.markdown("""
    Comparing phoneme characteristics across different categories, such as `phoneme_symbol`, `is_vowel`, or `dialect`, is essential for identifying distinct patterns and variations in speech sounds. This analysis helps us understand how individual phonemes behave, how vowels differ from consonants, and how regional accents might manifest acoustically. Such insights are crucial for developing a robust phoneme trainer or a highly accurate TTS system that can account for phonetic distinctions and regional variations.
    """)
    ```
*   **Code Stub (`plot_categorical_bar_comparison` function and calls)**:
    ```python
    def plot_categorical_bar_comparison(df, category_column, value_column, title, x_label, y_label):
        grouped_data = df.groupby(category_column)[value_column].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x=category_column, y=value_column, data=grouped_data, palette='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if len(grouped_data[category_column].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig

    if st.session_state.get('data_validated', False):
        st.subheader("Bar Charts for Categorical Comparisons")
        
        st.write("#### Average Phoneme Duration by Symbol")
        fig_b1 = plot_categorical_bar_comparison(
            st.session_state['phoneme_df'], 'phoneme_symbol', 'duration_ms', 
            'Average Phoneme Duration by Symbol', 'Phoneme Symbol', 'Average Duration (ms)'
        )
        st.pyplot(fig_b1)
        plt.close(fig_b1)

        st.write("#### Average Pitch by Dialect")
        fig_b2 = plot_categorical_bar_comparison(
            st.session_state['phoneme_df'], 'dialect', 'avg_pitch_hz', 
            'Average Pitch by Dialect', 'Dialect', 'Average Pitch (Hz)'
        )
        st.pyplot(fig_b2)
        plt.close(fig_b2)

        st.write("#### Average Naturalness Score: Vowels vs. Consonants")
        fig_b3 = plot_categorical_bar_comparison(
            st.session_state['phoneme_df'], 'is_vowel', 'pronunciation_naturalness_score', 
            'Average Naturalness Score: Vowels vs. Consonants', 'Is Vowel?', 'Average Naturalness Score'
        )
        st.pyplot(fig_b3)
        plt.close(fig_b3)

        st.markdown("""
        The bar plots provide clear comparisons across different categorical groups:

        *   **Average Phoneme Duration by Symbol**: This plot shows how the average `duration_ms` varies for each `phoneme_symbol`. We can observe that vowels (e.g., 'a', 'e', 'i', 'o', 'u') generally have longer average durations compared to consonants. This reflects the synthetic generation logic where vowels were designed to be more sustained.

        *   **Average Pitch by Dialect**: The plot comparing `avg_pitch_hz` across different `dialect` categories reveals subtle (synthetic) variations. For instance, 'General American' might have a slightly different average pitch compared to 'British English' or 'Australian English'. These differences, while synthetic, illustrate how regional variations can impact fundamental speech characteristics.

        *   **Average Naturalness Score: Vowels vs. Consonants**: This bar plot effectively shows whether `is_vowel` (True/False) correlates with the `pronunciation_naturalness_score`. Given our synthetic generation, vowels likely have a higher average naturalness score, as they are often more acoustically prominent and central to prosody. This comparison highlights how different types of phonemes might inherently contribute differently to perceived speech quality.

        These observations are critical for understanding how specific phoneme types and dialects contribute to the overall acoustic properties of speech. Such insights are directly applicable to a phoneme trainer, where learners could focus on specific phoneme categories or dialectal pronunciations based on their distinct characteristics.
        """)
    else:
        st.info("Perform data generation and validation first to visualize categorical comparisons.")
    ```

#### 4.8. Concept of Latent Features in TTS
This section introduces the theoretical concept of latent features in TTS systems.

*   **Markdown Content**:
    ```python
    st.header("7. Concept of Latent Features in TTS")
    st.markdown("""
    In advanced Text-to-Speech (TTS) systems, raw linguistic features like phonemes, pitch, energy, and duration are often too numerous and complex to be directly used by subsequent synthesis modules. This is where the concept of "**latent features**" (also known as embeddings or representations) becomes critical. As described in TTS architectures (e.g., the Encoder section of a TTS paper, page 2), an `Encoder` component transforms these high-dimensional, explicit `Linguistic features` into a lower-dimensional, more abstract representation called `Latent feature`.

    These latent features are not directly interpretable in the same way as raw features (e.g., "duration in ms"), but they are incredibly powerful because they capture the complex, underlying relationships and patterns within the speech characteristics. They are crucial for controlling the naturalness and expressiveness of the synthesized audio and serve as a compact, efficient input for the `Decoder` module, which then reconstructs the mel-spectrogram or other acoustic representations.

    This abstraction helps in several ways:
    *   **Dimensionality Reduction**: Reduces the number of variables the model needs to process.
    *   **Feature Compression**: Captures the most salient information in a compact form.
    *   **Complex Relationship Modeling**: Allows the model to learn and represent intricate, non-linear relationships between various acoustic properties.

    For the scope of this application, we will create a *simplified* representation of latent features by standardizing and combining existing numeric data. This will provide a conceptual understanding of how raw features can be transformed into a more abstract, model-friendly format.
    """)
    ```

#### 4.9. Simulating Phoneme Feature Embeddings
This section demonstrates the creation of simplified "latent features" through data standardization.

*   **Markdown Content**:
    ```python
    st.header("8. Simulating Phoneme Feature Embeddings")
    st.markdown("""
    To conceptually demonstrate the creation of "latent features" or "embeddings" from our raw phoneme characteristics, we will perform a simplified transformation. This process mimics the role of the `Encoder` in a TTS system, where raw, interpretable features are converted into a more abstract, compact representation. This abstraction is essential for machine learning models to effectively learn and process complex speech patterns.

    Our simulation will involve:
    1.  **Standardization**: Scaling our numeric features (duration, pitch, energy) to have a mean of 0 and a standard deviation of 1. This is crucial for many machine learning algorithms, as it prevents features with larger scales from dominating the learning process.
    2.  **Combination**: Creating a new DataFrame where these scaled features serve as our 'latent features'.

    The `StandardScaler` from `sklearn.preprocessing` is ideal for this task, as it transforms data $x$ using the formula:
    $$ z = \frac{x - \mu}{\sigma} $$
    where $\mu$ is the mean of the feature and $\sigma$ is its standard deviation.
    """)
    ```
*   **Code Stub (`create_simple_latent_features` function and calls)**:
    ```python
    def create_simple_latent_features(df, numeric_features_list):
        """
        Standardizes specified numeric features from the DataFrame using StandardScaler
        to create 'latent features'.
        """
        features_to_scale = df[numeric_features_list]
        scaler = StandardScaler()
        scaled_features_array = scaler.fit_transform(features_to_scale)
        scaled_df = pd.DataFrame(scaled_features_array, 
                                 columns=numeric_features_list, 
                                 index=df.index)
        return scaled_df

    if st.session_state.get('data_validated', False):
        st.subheader("Generated Latent Features")
        numeric_features = ['duration_ms', 'avg_pitch_hz', 'max_energy']
        st.session_state['latent_features_df'] = create_simple_latent_features(st.session_state['phoneme_df'], numeric_features)
        
        st.write("First 5 rows of the simulated latent features:")
        st.dataframe(st.session_state['latent_features_df'].head())
        st.markdown("""
        The output displays the first 5 rows of our `latent_features_df`. We can observe that the raw features (`duration_ms`, `avg_pitch_hz`, `max_energy`) have been transformed into a standardized scale. Each column now has a mean close to 0 and a standard deviation close to 1.

        This `latent_features_df` conceptually represents a simplified "embedding" or "latent feature" for each phoneme instance. By standardizing the features, we've removed their original scale and made them comparable, regardless of their initial units or ranges. This transformation is a common preprocessing step in machine learning, and it prepares these features for tasks such as clustering, where distance metrics are sensitive to feature scales.
        """)
    else:
        st.info("Perform data generation and validation first to create latent features.")
    ```

#### 4.10. Clustering Phonemes by Similar Characteristics
This section applies K-Means clustering to group phonemes based on their acoustic characteristics.

*   **Markdown Content**:
    ```python
    st.header("9. Clustering Phonemes by Similar Characteristics")
    st.markdown("""
    Clustering is an unsupervised machine learning technique that helps us identify natural groupings or segments within our data. In the context of phonetics, clustering phonemes by their acoustic characteristics can be incredibly useful. For language learners, it could help group similar-sounding phonemes that might be easily confused, aiding in targeted pronunciation practice. It also helps us understand inherent phonetic distinctions and similarities within our synthetic dataset.

    We will use the **K-Means algorithm** for clustering. K-Means is an iterative algorithm that partitions $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean (centroid). The algorithm aims to minimize the **within-cluster sum of squares (WCSS)**, which is a measure of the variability within each cluster. The objective function is defined as:

    $$ WCSS = \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2 $$

    where:
    *   $k$ is the number of clusters.
    *   $S_i$ is the $i$-th cluster.
    *   $x$ is a data point belonging to cluster $S_i$.
    *   $\mu_i$ is the centroid (mean) of cluster $S_i$.

    For this example, we will choose $k=4$ clusters to group our phonemes based on their `latent_features_df` (scaled `duration_ms`, `avg_pitch_hz`, `max_energy`).
    """)
    ```
*   **Code Stub (`perform_kmeans_clustering` function and calls)**:
    ```python
    def perform_kmeans_clustering(data, n_clusters, random_state):
        """
        Applies K-Means clustering to identify groups in data.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        kmeans.fit(data)
        return kmeans.labels_

    if 'latent_features_df' in st.session_state and st.session_state.get('data_validated', False):
        st.subheader("K-Means Clustering")
        num_clusters = st.slider(
            "Number of Clusters ($k$)", min_value=2, max_value=10, value=st.session_state.get('num_clusters', 4), step=1,
            help="Choose the number of clusters for K-Means algorithm. Default is 4."
        )
        if st.button("Perform Clustering", key="perform_clustering_button"):
            st.session_state['num_clusters'] = num_clusters
            with st.spinner(f"Clustering with {num_clusters} clusters..."):
                cluster_labels = perform_kmeans_clustering(
                    st.session_state['latent_features_df'], num_clusters, st.session_state['random_seed']
                )
                st.session_state['phoneme_df']['cluster_label'] = cluster_labels # Add to original df
            st.success("Clustering complete!")
        
        if 'cluster_label' in st.session_state['phoneme_df'].columns:
            st.write("Cluster counts:")
            st.dataframe(st.session_state['phoneme_df']['cluster_label'].value_counts())

            st.write("#### Phoneme Clusters: Duration vs. Pitch")
            fig_cluster, ax_cluster = plt.subplots(figsize=(10, 7))
            sns.scatterplot(
                data=st.session_state['phoneme_df'], 
                x='duration_ms', 
                y='avg_pitch_hz', 
                hue='cluster_label', 
                palette='viridis', 
                s=70, 
                alpha=0.8,
                ax=ax_cluster
            )
            ax_cluster.set_title('Phoneme Clusters: Duration vs. Pitch')
            ax_cluster.set_xlabel('Duration (ms)')
            ax_cluster.set_ylabel('Average Pitch (Hz)')
            ax_cluster.legend(title='Cluster')
            ax_cluster.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_cluster)
            plt.close(fig_cluster)

            st.markdown("""
            The clustering results and the scatter plot of 'Duration vs. Pitch' colored by `cluster_label` provide visual insights into how phonemes group based on their characteristics:

            *   **Cluster Counts**: The `value_counts()` output for `cluster_label` shows the distribution of phonemes across the `num_clusters` clusters. This indicates how balanced or unbalanced the groupings are.

            *   **Visual Interpretation**: The scatter plot visually demonstrates the separation of phonemes into distinct clusters. We can observe different regions of the plot corresponding to different cluster labels. For example:
                *   One cluster might primarily contain phonemes with shorter durations and lower pitches (likely consonants).
                *   Another cluster might encompass phonemes with longer durations and higher pitches (likely vowels).
                *   Intermediate clusters could represent phonemes with mixed characteristics or those falling on the boundaries.

            This clustering conceptually aids in categorizing phonemes for educational purposes. For instance, phonemes within the same cluster might be acoustically similar, making them 'difficult to distinguish' phoneme groups for language learners. Identifying such groups can help trainers focus on specific pronunciation challenges and provide targeted exercises.
            """)
    else:
        st.info("Perform data generation, validation, and latent feature creation first to perform clustering.")
    ```

#### 4.11. Introducing a Synthetic "Naturalness Score"
This section explains the target variable for modeling and prepares data for regression.

*   **Markdown Content**:
    ```python
    st.header("10. Introducing a Synthetic \"Naturalness Score\"")
    st.markdown("""
    In real-world Text-to-Speech (TTS) systems and language learning applications, the "naturalness" or "quality" of speech pronunciation is a critical metric. For our synthetic dataset, the `pronunciation_naturalness_score` serves as a proxy for this concept. It's a synthetic metric, ranging from 0 to 100, that quantifies how "natural" or "well-formed" a phoneme's pronunciation is based on its underlying acoustic characteristics (duration, pitch, energy). This concept is inspired by the discussions around the "naturalness of voice" and speech quality in phonetic and TTS research.

    This score will now serve as our **target variable** for a simple predictive modeling task. By building a model to predict this score from the acoustic features, we can conceptually understand how these features contribute to perceived naturalness and how a system might assess or even improve pronunciation quality.

    We will now prepare our data by selecting the relevant features ($X$) and our target variable ($y$), and then split it into training and testing sets to evaluate our model's performance rigorously.
    """)
    ```
*   **Code Stub (data split)**:
    ```python
    if 'phoneme_df' in st.session_state and st.session_state.get('data_validated', False):
        st.subheader("Data Preparation for Modeling")
        X = st.session_state['phoneme_df'][['duration_ms', 'avg_pitch_hz', 'max_energy']]
        y = st.session_state['phoneme_df']['pronunciation_naturalness_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=st.session_state['random_seed'])

        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

        st.write(f"Training set size: {len(X_train)} samples")
        st.write(f"Testing set size: {len(X_test)} samples")
        st.success("Features (X) and target (y) prepared and split into training/testing sets.")
        st.markdown("""
        The features ($X$) and the target variable ($y$) have been correctly identified from our `phoneme_df`. The dataset has been successfully split into training and testing sets, with 80% of the data allocated for training the model and 20% for evaluating its performance.

        This preparation ensures that our model will be trained on one subset of the data and then tested on unseen data, providing a more reliable assessment of its generalization capability. We are now ready to train a simple regression model to predict the `pronunciation_naturalness_score`.
        """)
    else:
        st.info("Perform data generation and validation first to prepare data for modeling.")
    ```

#### 4.12. Modeling Phoneme Naturalness
This section trains and evaluates a linear regression model to predict phoneme naturalness.

*   **Markdown Content**:
    ```python
    st.header("11. Modeling Phoneme Naturalness")
    st.markdown("""
    To understand how phoneme characteristics predict its perceived naturalness, we can employ a regression model. A regression model aims to establish a mathematical relationship between one or more independent variables (our phoneme characteristics) and a dependent variable (the naturalness score). This simulates how a system might quantitatively assess or even provide feedback to improve pronunciation.

    We will use **Linear Regression**, a simple yet powerful and interpretable model. Linear regression models the relationship between a dependent variable $y$ and one or more independent variables $X$ by fitting a linear equation to the observed data. The simple linear regression equation (for a single independent variable) is given by:

    $$ y = \beta_0 + \beta_1 x + \epsilon $$

    And for multiple independent variables (as in our case):

    $$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon $$

    Where:
    *   $y$ is the dependent variable (e.g., `pronunciation_naturalness_score`).
    *   $x_1, x_2, \dots, x_n$ are the independent variables (e.g., `duration_ms`, `avg_pitch_hz`, `max_energy`).
    *   $\beta_0$ is the y-intercept.
    *   $\beta_1, \beta_2, \dots, \beta_n$ are the coefficients (slopes) for each independent variable, indicating the change in $y$ for a one-unit change in $x_i$ while holding other variables constant.
    *   $\epsilon$ is the error term, representing the irreducible error in the model.

    After training, we will evaluate the model's performance using metrics like **R-squared** (proportion of variance explained) and **Mean Absolute Error (MAE)** (average absolute difference between predictions and actual values).
    """)
    ```
*   **Code Stub (`train_simple_regression_model`, `evaluate_regression_model` functions and calls)**:
    ```python
    def train_simple_regression_model(X_train, y_train):
        """
        Trains a sklearn.linear_model.LinearRegression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_regression_model(model, X_test, y_test):
        """
        Evaluates the performance of a trained regression model.
        """
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return r2, mae

    if all(key in st.session_state for key in ['X_train', 'y_train', 'X_test', 'y_test']):
        st.subheader("Linear Regression Model")
        if st.button("Train and Evaluate Regression Model", key="train_eval_model_button"):
            with st.spinner("Training model..."):
                regression_model = train_simple_regression_model(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['regression_model'] = regression_model
            st.success("Regression Model Training Complete.")

            st.write("### Model Coefficients:")
            for feature, coef in zip(st.session_state['X_train'].columns, regression_model.coef_):
                st.write(f"- {feature}: {coef:.4f}")
            st.write(f"- Intercept: {regression_model.intercept_:.4f}")

            st.write("### Evaluating Model Performance on Test Set:")
            r2, mae = evaluate_regression_model(regression_model, st.session_state['X_test'], st.session_state['y_test'])
            st.write(f"R-squared Score: {r2:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.markdown("""
            The linear regression model has been trained and evaluated:

            *   **Model Coefficients**: The coefficients indicate the synthetic influence of each feature on the `pronunciation_naturalness_score`:
                *   `duration_ms`: A positive coefficient suggests that, all else being equal, longer durations contribute positively to the naturalness score (within the simulated ideal range).
                *   `avg_pitch_hz`: A positive coefficient indicates that higher average pitch also positively influences the naturalness score.
                *   `max_energy`: A relatively larger positive coefficient implies that maximum energy has a strong synthetic positive impact on the naturalness score.
                *   `Intercept`: The baseline naturalness score when all features are zero (though in practice, features won't be zero).

            *   **R-squared Score**: A high R-squared score (e.g., close to 1) indicates that a large proportion of the variance in the `pronunciation_naturalness_score` can be explained by our chosen features. This means the model fits the synthetic data well.

            *   **Mean Absolute Error (MAE)**: The MAE tells us the average magnitude of the errors in a set of predictions, without considering their direction. A low MAE suggests that our model's predictions are, on average, close to the actual naturalness scores.

            These results confirm that, within our synthetic framework, phoneme characteristics like duration, pitch, and energy are strong predictors of pronunciation naturalness. This conceptually demonstrates how a TTS system or a phoneme trainer could assess and predict speech quality based on acoustic features.
            """)
    else:
        st.info("Prepare data for modeling first to train and evaluate the regression model.")
    ```

#### 4.13. Interactive Phoneme Analysis
This section provides an interactive tool for exploring and comparing individual phoneme characteristics.

*   **Markdown Content**:
    ```python
    st.header("12. Interactive Phoneme Analysis")
    st.markdown("""
    Interactive tools are invaluable for language learners and phoneticians alike. They provide a dynamic way to explore and compare phonemes, reinforcing auditory discrimination and phonetic awareness. By allowing users to interact directly with the data, they can gain a deeper, more intuitive understanding of how different acoustic characteristics contribute to the uniqueness of each sound.

    We will create an interactive display that enables users to select a `phoneme_symbol` from a dropdown menu. Upon selection, the application will dynamically display the average `duration_ms`, `avg_pitch_hz`, `max_energy`, and `pronunciation_naturalness_score` for that specific phoneme. Furthermore, it will generate a bar chart comparing these averages against the overall dataset averages, providing immediate context and highlighting distinctive features.
    """)
    ```
*   **Code Stub (Streamlit implementation of interactive analyzer)**:
    ```python
    if 'phoneme_df' in st.session_state and st.session_state.get('data_validated', False):
        st.subheader("Interactive Phoneme Analyzer")
        unique_phoneme_symbols = sorted(st.session_state['phoneme_df']['phoneme_symbol'].unique().tolist())
        
        selected_phoneme = st.selectbox(
            "Select a Phoneme:", 
            options=unique_phoneme_symbols,
            help="Choose a phoneme symbol to see its average characteristics and comparison."
        )

        if selected_phoneme:
            CHAR_COLS = ['duration_ms', 'avg_pitch_hz', 'max_energy', 'pronunciation_naturalness_score']
            
            overall_averages = st.session_state['phoneme_df'][CHAR_COLS].mean()
            phoneme_averages = st.session_state['phoneme_df'].groupby('phoneme_symbol')[CHAR_COLS].mean()
            
            selected_phoneme_data = phoneme_averages.loc[selected_phoneme]

            st.markdown(f"### Characteristics for Phoneme: '{selected_phoneme}'")
            for char_col in CHAR_COLS:
                val = selected_phoneme_data.get(char_col)
                if pd.isna(val):
                    st.write(f"**{char_col.replace('_', ' ').title()}:** N/A")
                else:
                    st.write(f"**{char_col.replace('_', ' ').title()}:** {val:.2f}")

            # Generate Bar Chart for comparison
            fig_compare, ax_compare = plt.subplots(figsize=(10, 6))

            labels = [col.replace('_', ' ').title() for col in CHAR_COLS[:-1]] # Exclude Naturalness Score from bar chart comparison
            x = np.arange(len(labels))
            width = 0.35

            phoneme_values_plot = [selected_phoneme_data.get(col, 0) for col in CHAR_COLS[:-1]]
            overall_values_plot = [overall_averages.get(col, 0) for col in CHAR_COLS[:-1]]

            ax_compare.bar(x - width/2, phoneme_values_plot, width, label=f'Phoneme "{selected_phoneme}"')
            ax_compare.bar(x + width/2, overall_values_plot, width, label='Overall Dataset')

            ax_compare.set_ylabel('Average Value')
            ax_compare.set_title(f'Comparison of Acoustic Characteristics for Phoneme "{selected_phoneme}" vs. Overall Dataset')
            ax_compare.set_xticks(x)
            ax_compare.set_xticklabels(labels, rotation=45, ha="right")
            ax_compare.legend()
            ax_compare.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_compare)
            plt.close(fig_compare)
        
        st.markdown("""
        The interactive display above allows you to dynamically explore the acoustic characteristics of different phonemes in our synthetic dataset. To use it:

        1.  **Select a Phoneme**: Use the dropdown menu to choose any `phoneme_symbol` (e.g., 'a', 's', 'k').
        2.  **Observe Characteristics**: Upon selection, the text output will immediately show the average `duration_ms`, `avg_pitch_hz`, `max_energy`, and `pronunciation_naturalness_score` for that specific phoneme.
        3.  **Compare with Overall Averages**: A bar chart will also be generated, visually comparing the selected phoneme's average acoustic features against the overall averages across the entire dataset. This helps highlight how a particular phoneme stands out (or doesn't) in terms of its characteristics.

        This interactive comparison is a practical tool in a conceptual phoneme trainer. For language learners, it allows for direct exploration of acoustic differences, making it easier to understand why certain sounds might be harder to distinguish or pronounce correctly. For example, you can see if a specific phoneme has a significantly longer duration or higher pitch compared to the average, indicating its unique acoustic signature.
        """)
    else:
        st.info("Perform data generation and validation first to use the interactive analyzer.")
    ```

#### 4.14. Conclusion and Key Insights
This section summarizes the key findings and insights from the entire application.

*   **Markdown Content**:
    ```python
    st.header("13. Conclusion and Key Insights")
    st.markdown("""
    Throughout this application, we've embarked on an interactive journey to explore the fundamental characteristics of phonemes within a synthetic dataset, drawing inspiration from the principles of phonetics and Text-to-Speech (TTS) systems. We've gained several key insights:

    *   **Phoneme Characteristic Distributions**: Our initial visualizations (histograms) revealed the varied distributions of features like `duration_ms`, `avg_pitch_hz`, and `max_energy`, often showing distinct patterns for vowels and consonants. This underscores the diverse acoustic nature of different speech sounds.

    *   **Relationships Between Features**: Scatter plots and pair plots elucidated the interrelationships between these acoustic features. We observed that longer durations often correlate with higher energy, and that categorical factors like `is_vowel` and `dialect` significantly influence pitch, duration, and energy profiles. These correlations are critical for modeling complex speech patterns.

    *   **Categorical Differences**: Bar plots highlighted clear distinctions in average characteristics across `phoneme_symbol` and `dialect`, and between vowels and consonants. This demonstrates how specific sounds and regional variations possess unique acoustic signatures.

    *   **Concept of Latent Features and Clustering**: We simulated the creation of "latent features" through standardization, conceptually mirroring how TTS encoders transform raw features into abstract representations. K-Means clustering then allowed us to group phonemes by similar characteristics, illustrating how such techniques could categorize phonemes for targeted language learning.

    *   **Modeling Phoneme Naturalness**: A simple linear regression model successfully predicted a synthetic `pronunciation_naturalness_score` from acoustic features. This demonstrated how data-driven approaches can quantify and assess speech quality, a concept central to improving TTS output and providing feedback in phoneme trainers.

    *   **Interactive Exploration**: The interactive phoneme analyzer showcased the power of Streamlit to create engaging tools for language learners, allowing dynamic comparison of phoneme characteristics against overall averages. This fosters deeper phonetic awareness and supports targeted practice.

    In conclusion, this exploration reinforces how a granular understanding of phoneme characteristics—duration, pitch, and energy—is foundational to both the generation of natural-sounding synthetic speech and the development of effective language learning tools. Data-driven approaches, from visualization to predictive modeling and interactive analysis, offer immense potential for advancing phonetic awareness and speech technology. The insights gained here are directly relevant to the idea of an "Interactive Phoneme Trainer," providing the analytical backbone for such an application.
    """)
    ```

#### 4.15. References
This section provides a list of academic references cited or relevant to the application's content.

*   **Markdown Content**:
    ```python
    st.header("14. References")
    st.markdown("""
    *   Chowdhury, Md. Jalal Uddin, and Ashab Hussan. "A review-based study on different Text-to-Speech technologies." *International Journal of Computer Science and Network Security* 19.3 (2019): 173-181.
    *   Sadeque, F. Y., Yasar, S., & Islam, M. M. (2013, May). Bangla text to speech conversion: A syllabic unit selection approach. In 2013 International Conference on Informatics, Electronics and Vision (ICIEV) (pp. 1-6). IEEE.
    *   Alam, Firoj, Promila Kanti Nath, & Khan, Mumit (2007). 'Text to speech for Bangla language using festival'. BRAC University.
    *   Alam, Firoj, Promila Kanti Nath, & Khan, Mumit (2011). 'Bangla text to speech using festival', Conference on human language technology for development, pp.154-161.
    *   Arafat, M. Y., Fahrin, S., Islam, M. J., Siddiquee, M. A., Khan, A., Kotwal, M. R. A., & Huda, M. N. (2014, December). Speech synthesis for Bangla text to speech conversion. In The 8th International Conference on Software, Knowledge, Information Management and Applications (SKIMA 2014) (pp. 1-6). IEEE.
    *   Ahmed, K. M., Mandal, P., & Hossain, B. M. (2019). Text to Speech Synthesis for Bangla Language. International Journal of Information Engineering and Electronic Business, 12(2), 1.
    *   Łańcucki, A. (2021). "Fastpitch: Parallel Text-to-Speech with Pitch Prediction." *ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 6588-6592. doi: 10.1109/ICASSP39728.2021.9413889.
    *   Luo, R., et al. (2021). "Lightspeech: Lightweight and Fast Text to Speech with Neural Architecture Search." *ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 5699-5703. doi: 10.1109/ICASSP39728.2021.9414403.
    *   Alam, Firoj, S. M. Murtoza Habib, & Khan, Mumit (2009). "Text normalization system for Bangla," Proc. of Conf. on Language and Technology, Lahore, pp. 22-24.
    *   Berk, Elliot (2004). JFlex - The Fast Scanner Generator for Java, version 1.4.1. [http://jflex.de](http://jflex.de)
    *   Tran, D., Haines, P., Ma, W., & Sharma, D. (2007, September). Text-to-speech technology-based programming tool. In International Conference On Signal, Speech and Image Processing.
    *   Rashid, M. M., Hussain, M. A., & Rahman, M. S. (2010). Text normalization and diphone preparation for Bangla speech synthesis. Journal of Multimedia, 5(6), 551.
    *   Zeki, M., Khalifa, O. O., & Naji, A. W. (2010, May). Development of an Arabic text-to-speech system. In International Conference on Computer and Communication Engineering (ICCCE'10) (pp. 1-5). IEEE.
    """)
    ```

