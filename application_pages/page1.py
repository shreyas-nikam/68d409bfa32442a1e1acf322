import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error 
import io

# Configure Seaborn for better aesthetics and colorblind-friendly palette
sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def run_page1():
    st.title("Phoneme Characteristics Analysis: An Interactive Exploration for Text-to-Speech Understanding")
    st.markdown("""
    This application delves into the fascinating world of **phonetics**, the scientific study of speech sounds, and its crucial role in **Text-to-Speech (TTS)** technology. TTS systems, which convert written text into audible speech, are becoming increasingly sophisticated, powering virtual assistants, accessibility tools, and various interactive applications.

    At the heart of speech are **phonemes**: \"The smallest unit of sound that makes a word's pronunciation and meaning different from another word.\" These fundamental sound units are not just abstract concepts; they possess measurable characteristics like **duration**, **pitch**, and **energy**, which are vital for producing natural-sounding speech. These features contribute significantly to **prosody**, the rhythm, stress, and intonation of speech, making the generated voice expressive and intelligible.

    In a typical TTS system, as illustrated in many research papers (e.g., Fig. 1 in \"A review-based study on different Text-to-Speech technologies\"), components like the preprocessor, encoder, decoder, and vocoder work in concert. The encoder, for instance, often takes linguistic features, including phonemes and their characteristics, and transforms them into more abstract \"latent features\" that the decoder then uses to generate mel-spectrograms (visual representations of sound frequencies over time), ultimately leading to the synthesized speech.

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
    
    st.header("2. Exploring the Synthetic Phoneme Data")
    st.markdown("""
    Initial data exploration is a crucial step to understand the structure, content, and basic statistics of our dataset. This helps us ensure that the synthetic data generation process has yielded results that align with our expectations for further analysis related to phoneme characteristics. We'll examine data types, non-null counts, and summary statistics for numeric columns.
    """)
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
    
    st.header("3. Data Validation and Summary Statistics")
    st.markdown("""
    Data validation is a critical step to ensure the quality and reliability of our dataset before proceeding with any in-depth analysis or modeling. By systematically checking for expected column names, verifying data types, and asserting the absence of missing values in `critical_fields`, we can prevent downstream errors and ensure the integrity of our phoneme characteristic data.

    For this dataset, we consider `duration_ms`, `avg_pitch_hz`, and `max_energy` as `critical_fields` because these are fundamental acoustic properties that must be present and valid for any meaningful phonetic analysis.
    """)

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
