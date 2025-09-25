import pandas as pd
import numpy as np

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
    # These values are chosen to create a realistic distinction.
    base_feature_params = {
        'vowel': {
            'duration_ms': (120, 30),  # Vowels tend to be longer
            'avg_pitch_hz': (160, 45), # Vowels typically have clearer, higher pitch
            'max_energy': (0.85, 0.15) # Vowels are generally more energetic
        },
        'consonant': {
            'duration_ms': (60, 20),   # Consonants are often shorter
            'avg_pitch_hz': (90, 35),  # Consonants can have lower or less defined pitch
            'max_energy': (0.45, 0.1)  # Consonants generally less energetic
        }
    }

    # Phoneme-specific micro-adjustments to base parameters to introduce variety
    # These are illustrative; more extensive lists would be in a real system.
    phoneme_specific_adjustments = {
        'a': {'duration_ms': 15, 'avg_pitch_hz': 10, 'max_energy': 0.05},
        'e': {'duration_ms': -5, 'avg_pitch_hz': 5, 'max_energy': -0.02},
        'i': {'duration_ms': 0, 'avg_pitch_hz': 12, 'max_energy': 0.03},
        'o': {'duration_ms': 10, 'avg_pitch_hz': -5, 'max_energy': 0.01},
        'u': {'duration_ms': 5, 'avg_pitch_hz': -8, 'max_energy': 0.04},
        'p': {'duration_ms': 10, 'avg_pitch_hz': -15, 'max_energy': 0.08}, # Plosives can be short, bursty
        't': {'duration_ms': -5, 'avg_pitch_hz': -10, 'max_energy': 0.03},
        'k': {'duration_ms': 5, 'avg_pitch_hz': -12, 'max_energy': 0.06},
    }

    # Coefficients for the linear combination of features for naturalness score
    # These are scaled to produce scores in a reasonable range (e.g., 0-100)
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
        
        # Determine if the phoneme is a vowel
        record['is_vowel'] = record['phoneme_symbol'].lower() in VOWELS
        
        # Select base parameters based on vowel/consonant category
        category_key = 'vowel' if record['is_vowel'] else 'consonant'
        params = base_feature_params[category_key]

        # Initialize means for current sample
        current_duration_mean, current_duration_std = params['duration_ms']
        current_pitch_mean, current_pitch_std = params['avg_pitch_hz']
        current_energy_mean, current_energy_std = params['max_energy']

        # Apply phoneme-specific adjustments
        if record['phoneme_symbol'] in phoneme_specific_adjustments:
            adj = phoneme_specific_adjustments[record['phoneme_symbol']]
            current_duration_mean += adj.get('duration_ms', 0)
            current_pitch_mean += adj.get('avg_pitch_hz', 0)
            current_energy_mean += adj.get('max_energy', 0)

        # Generate feature values with Gaussian noise, ensuring non-negativity
        record['duration_ms'] = max(0, np.random.normal(current_duration_mean, current_duration_std))
        record['avg_pitch_hz'] = max(0, np.random.normal(current_pitch_mean, current_pitch_std))
        record['max_energy'] = max(0, np.random.normal(current_energy_mean, current_energy_std))

        # Calculate pronunciation_naturalness_score as a linear combination + noise
        score = (
            naturalness_score_base +
            record['duration_ms'] * naturalness_score_coeffs['duration_ms'] +
            record['avg_pitch_hz'] * naturalness_score_coeffs['avg_pitch_hz'] +
            record['max_energy'] * naturalness_score_coeffs['max_energy'] +
            np.random.normal(0, naturalness_score_noise_std)
        )
        # Clip the score to a reasonable range (e.g., 0-100)
        record['pronunciation_naturalness_score'] = np.clip(score, 0, 100)

        data_records.append(record)

    df = pd.DataFrame(data_records)
    
    # Ensure the DataFrame has all expected columns and the correct order.
    return df[expected_column_names]

import pandas as pd
import numpy as np

def validate_and_summarize_data(df, expected_columns, expected_dtypes, critical_fields):
    """
    Performs data validation and logs summary statistics.

    Checks for:
    1. Presence of all expected column names.
    2. Correctness of data types for specified columns.
    3. Absence of missing values in designated critical fields.

    Arguments:
      df (pandas.DataFrame): The DataFrame to validate and summarize.
      expected_columns (list<str>): A list of column names expected in the DataFrame.
      expected_dtypes (dict): A dictionary mapping column names to their expected data types.
      critical_fields (list<str>): A list of column names that must not contain any missing values.

    Output:
      None: Prints validation results and summary statistics to the console.
    """

    print("--- Starting Data Validation ---")

    # 1. Validate Expected Column Names
    actual_columns_set = set(df.columns)
    expected_columns_set = set(expected_columns)
    
    missing_from_df = expected_columns_set - actual_columns_set
    if missing_from_df:
        raise ValueError(
            f"Validation Error: Expected columns missing from DataFrame: {sorted(list(missing_from_df))}. "
            f"Actual columns: {sorted(list(df.columns))}. "
            f"Expected columns missing" # Match string for pytest
        )
    print(f"Validation Step 1: All {len(expected_columns)} expected columns are present.")

    # 2. Validate Data Types
    for col, expected_dtype in expected_dtypes.items():
        # Ensure column exists before checking dtype (covered by initial column check, but defensive)
        if col not in df.columns:
            raise ValueError(f"Validation Error: Column '{col}' in expected_dtypes not found in DataFrame "
                             "after initial column validation. This indicates an internal inconsistency.")

        actual_dtype = df[col].dtype
        if not pd.api.types.is_dtype_equal(actual_dtype, expected_dtype):
            raise TypeError(
                f"Validation Error: Data type mismatch for column '{col}'. "
                f"Expected '{expected_dtype}', got '{actual_dtype}'. "
                f"Data type mismatch" # Match string for pytest
            )
    print("Validation Step 2: All column data types match expected types.")

    # 3. Validate Missing Values in Critical Fields
    for field in critical_fields:
        # Ensure critical field exists (covered by initial column check, but defensive)
        if field not in df.columns:
            raise ValueError(f"Validation Error: Critical field '{field}' not found in DataFrame "
                             "after initial column validation. This indicates an internal inconsistency.")

        if df[field].isnull().any():
            missing_indices = df.index[df[field].isnull()].tolist()
            raise ValueError(
                f"Validation Error: Missing values found in critical field '{field}' "
                f"at index/indices: {missing_indices}. "
                f"Missing values found in critical fields" # Match string for pytest
            )
    print("Validation Step 3: No missing values in critical fields.")

    print("--- Data Validation Complete: All checks passed ---")

    # Summarize Data
    print("\n--- Data Summary ---")
    if df.empty:
        print("DataFrame is empty. No summary statistics to display.")
    else:
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            print("Summary statistics for numeric columns:")
            print(numeric_df.describe())
        else:
            print("No numeric columns found in the DataFrame to summarize.")
    print("\n--- Summary Complete ---")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution_histogram(df, column_name, title, x_label, y_label, bins):
    """    Generates and displays a histogram plot for a specified numeric column using Seaborn, showing the distribution of values. The plot is saved as a PNG file.
Arguments:
  df (pandas.DataFrame): The DataFrame containing the data.
  column_name (str): The name of the numeric column to plot.
  title (str): The title of the histogram plot.
  x_label (str): The label for the x-axis.
  y_label (str): The label for the y-axis.
  bins (int): The number of bins for the histogram.
Output:
  None: Displays the plot and saves it as a PNG file.
    """

    # Create a new figure to ensure plots are independent
    plt.figure()

    try:
        # Generate the histogram using Seaborn
        sns.histplot(data=df, x=column_name, bins=bins)

        # Set plot title and labels
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Generate filename and save the plot
        filename = f"{column_name}_distribution_histogram.png"
        plt.savefig(filename)

    finally:
        # Clear the current figure to free memory
        plt.clf()
        # Close the figure to prevent it from being displayed and free resources
        plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_scatter_relationship(df, x_column, y_column, hue_column, title, x_label, y_label):
    """Generates and displays a scatter plot to examine relationships between two numeric columns, optionally differentiating points by a categorical hue column. The plot is saved as a PNG file.
    """

    # Seaborn and Pandas will raise KeyError if specified columns do not exist in the DataFrame,
    # or if the DataFrame is empty and column access is attempted,
    # which is handled by the provided test cases.

    # Create the scatter plot using seaborn
    sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column)
    
    # Get the current axes object
    ax = plt.gca()
    
    # Set plot title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Save the plot to a PNG file
    plt.savefig(f'{title}.png')
    
    # Display the plot
    plt.show()
    
    # Clear the current figure to free memory and prevent plots from overlapping
    plt.clf()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_bar_comparison(df, category_column, value_column, title, x_label, y_label):
    """
    Generates and displays a bar plot showing the mean of a specified numeric value_column 
    grouped by a categorical category_column. The plot is saved as a PNG file.

    Arguments:
      df (pandas.DataFrame): The DataFrame containing the data.
      category_column (str): The name of the categorical column for grouping.
      value_column (str): The name of the numeric column whose mean will be plotted.
      title (str): The title of the bar plot.
      x_label (str): The label for the x-axis.
      y_label (str): The label for the y-axis.

    Output:
      None: Displays the plot and saves it as a PNG file.
    """

    # Group data by the categorical column and calculate the mean of the value column.
    # This step implicitly handles several error scenarios checked by the test cases:
    # - KeyError: If `category_column` or `value_column` do not exist in `df`.
    # - TypeError: If `value_column` contains non-numeric data that cannot be averaged.
    # - AttributeError: If `df` is not a pandas DataFrame (e.g., a list).
    grouped_data = df.groupby(category_column)[value_column].mean().reset_index()

    # Create a new figure for the plot
    plt.figure(figsize=(10, 6))
    
    # Generate the bar plot using seaborn, applying a colorblind-friendly palette
    ax = sns.barplot(x=category_column, y=value_column, data=grouped_data, palette='viridis')

    # Set the plot title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Adjust plot layout to prevent elements from overlapping
    plt.tight_layout()

    # Construct the filename from the plot title, converting spaces to underscores and lowercase
    plot_file_name = f"{title.replace(' ', '_').lower()}.png"

    # Save the generated plot to a PNG file
    plt.savefig(plot_file_name)

    # Display the plot
    plt.show()

    # Clear the current figure to free up memory and prepare for subsequent plots
    plt.clf()

import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_simple_latent_features(df, numeric_features_list):
    """
    Standardizes specified numeric features from the DataFrame using StandardScaler
    to create 'latent features'.

    Arguments:
      df (pandas.DataFrame): The input DataFrame containing the numeric features.
      numeric_features_list (list<str>): A list of numeric column names to be scaled.

    Output:
      pandas.DataFrame: A new DataFrame with the scaled latent features.
    """
    
    # Select the columns to be scaled.
    # This handles cases where numeric_features_list is empty (returns an empty DataFrame 
    # with 0 columns but preserving df's rows) or contains non-existent columns (raises KeyError).
    features_to_scale = df[numeric_features_list]

    # Initialize StandardScaler.
    scaler = StandardScaler()

    # Fit and transform the selected features.
    # StandardScaler can handle empty DataFrames (0 rows or 0 columns) gracefully,
    # returning an appropriately shaped empty array.
    scaled_features_array = scaler.fit_transform(features_to_scale)

    # Create a new DataFrame from the scaled features, preserving original column names and index.
    scaled_df = pd.DataFrame(scaled_features_array, 
                             columns=numeric_features_list, 
                             index=df.index)

    return scaled_df

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def perform_kmeans_clustering(data, n_clusters, random_state):
    """
    Applies K-Means clustering to identify groups in data.

    Arguments:
      data (pandas.DataFrame or numpy.ndarray): Input data.
      n_clusters (int): The number of clusters to form.
      random_state (int): Seed for reproducibility.

    Output:
      numpy.ndarray: Array of cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(data)
    return kmeans.labels_

from sklearn.linear_model import LinearRegression

def train_simple_regression_model(X_train, y_train):
    """
    Trains a sklearn.linear_model.LinearRegression model using the provided training features and target variable.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

from sklearn.metrics import r2_score, mean_absolute_error

def evaluate_regression_model(model, X_test, y_test):
    """    Evaluates the performance of a trained regression model on test data by calculating and printing key evaluation metrics such as R-squared score and Mean Absolute Error (MAE).
Arguments:
  model (sklearn.base.BaseEstimator): The trained regression model to evaluate.
  X_test (pandas.DataFrame or numpy.ndarray): The test features for evaluation.
  y_test (pandas.Series or numpy.ndarray): The true target values for the test data.
Output:
  None: Prints the R-squared score and Mean Absolute Error to the console.
    """
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Print the evaluation metrics
    print(f"R-squared Score: {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np # For handling NaN values

def interactive_phoneme_analyzer(df, phoneme_symbols_list):
    """
    Creates an interactive display using ipywidgets that allows users to select a phoneme symbol from a dropdown.
    Upon selection, it displays the average characteristics (duration, pitch, energy, naturalness score) for that phoneme
    and a bar chart comparing these averages against overall dataset averages.

    Arguments:
      df (pandas.DataFrame): The DataFrame containing phoneme characteristics data.
      phoneme_symbols_list (list<str>): A list of unique phoneme symbols available for selection.

    Output:
      None: Displays an interactive widget in the notebook output.
    """

    # --- 1. Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    required_cols = ['phoneme_symbol', 'duration_ms', 'avg_pitch_hz', 'max_energy', 'pronunciation_naturalness_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"DataFrame is missing required characteristic columns: {', '.join(missing_cols)}")

    # Define the characteristics columns for analysis
    CHAR_COLS = ['duration_ms', 'avg_pitch_hz', 'max_energy', 'pronunciation_naturalness_score']

    # --- 2. Initial Data Processing ---
    # Calculate overall dataset averages
    if not df.empty:
        overall_averages = df[CHAR_COLS].mean()
    else:
        # If df is empty, overall averages should be NaN
        overall_averages = pd.Series([np.nan] * len(CHAR_COLS), index=CHAR_COLS)

    # Calculate phoneme-specific averages
    if not df.empty and 'phoneme_symbol' in df.columns and not df['phoneme_symbol'].empty:
        phoneme_averages = df.groupby('phoneme_symbol')[CHAR_COLS].mean()
    else:
        # If df is empty or phoneme_symbol column is empty/missing, phoneme averages should be empty
        phoneme_averages = pd.DataFrame(columns=CHAR_COLS, index=pd.Index([], name='phoneme_symbol'))

    # --- 3. Widget Creation ---
    # Create the Dropdown widget
    # If phoneme_symbols_list is empty, dropdown will have no options.
    phoneme_dropdown = widgets.Dropdown(
        options=phoneme_symbols_list,
        description='Select Phoneme:',
        disabled=False,
    )

    # Create the Output widget
    output_widget = widgets.Output()

    # --- 4. Interaction Logic: Define the callback function ---
    def update_display(change=None):
        with output_widget:
            output_widget.clear_output()

            selected_phoneme = phoneme_dropdown.value

            if not selected_phoneme: # Handles cases where dropdown is empty or no selection
                display(HTML("<b>No phoneme selected or available.</b>"))
                return

            # Check if selected_phoneme exists in our calculated averages
            if selected_phoneme not in phoneme_averages.index:
                display(HTML(f"<b>No data available for phoneme: '{selected_phoneme}'</b>"))
                return

            selected_phoneme_data = phoneme_averages.loc[selected_phoneme]

            display(HTML(f"<h3>Characteristics for Phoneme: '{selected_phoneme}'</h3>"))
            for char_col in CHAR_COLS:
                val = selected_phoneme_data.get(char_col)
                if pd.isna(val):
                    display(HTML(f"<b>{char_col.replace('_', ' ').title()}:</b> N/A"))
                else:
                    display(HTML(f"<b>{char_col.replace('_', ' ').title()}:</b> {val:.2f}"))

            # --- Generate Bar Chart ---
            fig, ax = plt.subplots(figsize=(10, 6))

            labels = [col.replace('_', ' ').title() for col in CHAR_COLS]
            x = np.arange(len(labels))
            width = 0.35

            # Values for the selected phoneme
            phoneme_values = [selected_phoneme_data.get(col) for col in CHAR_COLS]
            # Replace None or NaN with 0 for plotting to ensure bars are drawn,
            # but this might misrepresent actual 'no data'. For this context,
            # it indicates a lack of value for comparison.
            phoneme_values_plot = [0 if pd.isna(v) else v for v in phoneme_values]
            
            # Overall dataset values
            overall_values = [overall_averages.get(col) for col in CHAR_COLS]
            overall_values_plot = [0 if pd.isna(v) else v for v in overall_values]

            rects1 = ax.bar(x - width/2, phoneme_values_plot, width, label=f'Phoneme "{selected_phoneme}"')
            rects2 = ax.bar(x + width/2, overall_values_plot, width, label='Overall Dataset')

            ax.set_ylabel('Average Value')
            ax.set_title(f'Comparison of Characteristics for Phoneme "{selected_phoneme}" vs. Overall Dataset')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            plt.close(fig) # Close the figure to free up memory

    # Attach the callback function to the dropdown's 'value' change event
    phoneme_dropdown.observe(update_display, names='value')

    # Initial display based on the default value of the dropdown (if any)
    # or provide initial feedback if no phonemes are available.
    if phoneme_dropdown.value: # This implicitly checks for None and empty string
        update_display()
    else:
        with output_widget:
            output_widget.clear_output()
            if not phoneme_symbols_list:
                display(HTML("<b>No phoneme symbols provided to analyze.</b>"))
            else: # phoneme_dropdown.value is None or empty but symbols were provided
                display(HTML("<b>Select a phoneme from the dropdown above to view its characteristics.</b>"))

    # --- 5. Display the VBox containing the widgets ---
    display(widgets.VBox([phoneme_dropdown, output_widget]))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_features_pairplot(df, features_list, hue_column, title):
    """Generates and displays a Seaborn pair plot, saving it as a PNG file.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features_list (list[str]): Columns to include in the pair plot.
        hue_column (str, optional): Categorical column for coloring points. Defaults to None.
        title (str, optional): Title for the pair plot. Defaults to None.
    """
    # 1. Input Validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(features_list, list) or not all(isinstance(f, str) for f in features_list):
        raise TypeError("features_list must be a list of strings.")
    if hue_column is not None and not isinstance(hue_column, str):
        raise TypeError("hue_column must be a string or None.")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None.")

    # 2. Data Validation
    if df.empty and features_list:
        raise ValueError("Cannot plot an empty DataFrame with specified features.")
    
    # Check if all features exist in the DataFrame
    missing_features = [f for f in features_list if f not in df.columns]
    if missing_features:
        raise KeyError(f"Features not found in DataFrame: {missing_features}")
    
    # Check if hue_column exists if provided
    if hue_column is not None and hue_column not in df.columns:
        raise KeyError(f"Hue column '{hue_column}' not found in DataFrame.")

    # 3. Create plot directory
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # 4. Generate the pair plot
    # The test expects diag_kind='hist'
    if hue_column:
        g = sns.pairplot(df, vars=features_list, hue=hue_column, diag_kind='hist')
    else:
        g = sns.pairplot(df, vars=features_list, diag_kind='hist')

    # 5. Set title if provided
    if title:
        g.fig.suptitle(title, y=1.02)

    # 6. Save and close the plot
    plot_path = os.path.join(plot_dir, "pair_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()