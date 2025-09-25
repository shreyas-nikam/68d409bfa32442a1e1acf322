import pytest
import pandas as pd
import numpy as np
from definition_ab682c9b840047c88fc19fc1abbc2ab6 import validate_and_summarize_data

# Define common expected_columns, expected_dtypes, critical_fields for testing
# These reflect typical data structures found in the notebook's context (e.g., phoneme data)
COMMON_EXPECTED_COLUMNS = ['phoneme_id', 'duration_ms', 'avg_pitch_hz', 'is_vowel']
COMMON_EXPECTED_DTYPES = {
    'phoneme_id': np.int64,
    'duration_ms': np.float64,
    'avg_pitch_hz': np.float64,
    'is_vowel': bool
}
COMMON_CRITICAL_FIELDS = ['duration_ms', 'avg_pitch_hz']

def test_valid_data_frame():
    """
    Test with a DataFrame that fully meets all validation criteria:
    correct columns, data types, and no missing values in critical fields.
    """
    df = pd.DataFrame({
        'phoneme_id': [1, 2, 3],
        'duration_ms': [85.5, 120.1, 95.0],
        'avg_pitch_hz': [150.2, 210.0, 145.8],
        'is_vowel': [True, False, True]
    })
    # No exception should be raised for valid data
    validate_and_summarize_data(df, COMMON_EXPECTED_COLUMNS, COMMON_EXPECTED_DTYPES, COMMON_CRITICAL_FIELDS)

def test_missing_expected_column():
    """
    Test with a DataFrame where one of the `expected_columns` is entirely missing.
    Should raise a ValueError indicating missing columns.
    """
    df = pd.DataFrame({
        'phoneme_id': [1, 2, 3],
        'duration_ms': [85.5, 120.1, 95.0],
        'is_vowel': [True, False, True]
    }) # 'avg_pitch_hz' is missing from COMMON_EXPECTED_COLUMNS
    with pytest.raises(ValueError, match="Expected columns missing"):
        validate_and_summarize_data(df, COMMON_EXPECTED_COLUMNS, COMMON_EXPECTED_DTYPES, COMMON_CRITICAL_FIELDS)

def test_incorrect_data_type_for_column():
    """
    Test with a DataFrame where a column has an incorrect data type compared
    to `expected_dtypes`. For example, a float column provided as strings.
    Should raise a TypeError for data type mismatch.
    """
    df = pd.DataFrame({
        'phoneme_id': [1, 2, 3],
        'duration_ms': ['85.5', '120.1', '95.0'], # Expected float64, got object/string
        'avg_pitch_hz': [150.2, 210.0, 145.8],
        'is_vowel': [True, False, True]
    })
    with pytest.raises(TypeError, match="Data type mismatch"):
        validate_and_summarize_data(df, COMMON_EXPECTED_COLUMNS, COMMON_EXPECTED_DTYPES, COMMON_CRITICAL_FIELDS)

def test_missing_values_in_critical_field():
    """
    Test with a DataFrame containing NaN values in a column designated as a
    `critical_field`.
    Should raise a ValueError indicating missing values in critical fields.
    """
    df = pd.DataFrame({
        'phoneme_id': [1, 2, 3],
        'duration_ms': [85.5, np.nan, 95.0], # NaN in 'duration_ms', which is critical
        'avg_pitch_hz': [150.2, 210.0, 145.8],
        'is_vowel': [True, False, True]
    })
    with pytest.raises(ValueError, match="Missing values found in critical fields"):
        validate_and_summarize_data(df, COMMON_EXPECTED_COLUMNS, COMMON_EXPECTED_DTYPES, COMMON_CRITICAL_FIELDS)

def test_empty_dataframe_with_valid_structure():
    """
    Test with an empty DataFrame that still adheres to the `expected_columns`
    and `expected_dtypes` structure.
    No exceptions should be raised as the structure is valid and there are no
    missing critical values (since there are no values at all).
    """
    # Create an empty DataFrame with the specified columns and dtypes
    df = pd.DataFrame(columns=COMMON_EXPECTED_COLUMNS)
    for col, dtype in COMMON_EXPECTED_DTYPES.items():
        if col in df.columns: # Ensure the column exists before casting (important for truly empty DF)
            df[col] = df[col].astype(dtype)

    # No exception should be raised as the empty DF structurally meets requirements
    validate_and_summarize_data(df, COMMON_EXPECTED_COLUMNS, COMMON_EXPECTED_DTYPES, COMMON_CRITICAL_FIELDS)