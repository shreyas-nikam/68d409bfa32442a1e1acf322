import pytest
import pandas as pd
from definition_0c350c29e3284ce9aa4052c3512ce9cc import generate_synthetic_phoneme_data

# Define typical inputs for tests
PHONEME_SYMBOLS = ['a', 'e', 'i', 'o', 'u', 'p', 't', 'k']
WORD_CONTEXTS = ['start', 'middle', 'end']
DIALECTS = ['US', 'UK', 'AUS']
RANDOM_SEED = 42

@pytest.mark.parametrize(
    "num_samples, phoneme_symbols, word_contexts, dialects, random_seed, expected_exception, expected_rows, expected_cols_present",
    [
        # Test Case 1: Basic functionality - successful generation with positive num_samples
        (10, PHONEME_SYMBOLS, WORD_CONTEXTS, DIALECTS, RANDOM_SEED, None, 10, True),
        
        # Test Case 2: Edge case - num_samples = 0, should return an empty DataFrame with columns
        (0, PHONEME_SYMBOLS, WORD_CONTEXTS, DIALECTS, RANDOM_SEED, None, 0, True),
        
        # Test Case 3: Edge case - empty phoneme_symbols list when num_samples > 0 (critical input missing)
        (5, [], WORD_CONTEXTS, DIALECTS, RANDOM_SEED, ValueError, None, False),
        
        # Test Case 4: Error case - num_samples is a non-integer type
        ("invalid", PHONEME_SYMBOLS, WORD_CONTEXTS, DIALECTS, RANDOM_SEED, TypeError, None, False),
        
        # Test Case 5: Error case - phoneme_symbols is not a list
        (10, "not_a_list_type", WORD_CONTEXTS, DIALECTS, RANDOM_SEED, TypeError, None, False),
    ]
)
def test_generate_synthetic_phoneme_data(num_samples, phoneme_symbols, word_contexts, dialects, random_seed, 
                                          expected_exception, expected_rows, expected_cols_present):
    # Define expected column names for a valid DataFrame output
    expected_column_names = [
        'phoneme_id', 'phoneme_symbol', 'word_context', 'duration_ms', 'avg_pitch_hz', 
        'max_energy', 'is_vowel', 'dialect', 'pronunciation_naturalness_score'
    ]

    if expected_exception:
        with pytest.raises(expected_exception):
            generate_synthetic_phoneme_data(num_samples, phoneme_symbols, word_contexts, dialects, random_seed)
    else:
        df = generate_synthetic_phoneme_data(num_samples, phoneme_symbols, word_contexts, dialects, random_seed)
        
        assert isinstance(df, pd.DataFrame), "Output should be a Pandas DataFrame."
        assert len(df) == expected_rows, f"DataFrame should have {expected_rows} rows."
        
        if expected_cols_present:
            assert list(df.columns) == expected_column_names, "DataFrame columns do not match expected names."
            
            if expected_rows > 0:
                # Check data types and non-null for critical columns
                numeric_cols = ['duration_ms', 'avg_pitch_hz', 'max_energy', 'pronunciation_naturalness_score']
                string_cols = ['phoneme_symbol', 'word_context', 'dialect']
                
                for col in numeric_cols:
                    assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' should be numeric."
                    assert not df[col].isnull().any(), f"Column '{col}' should not contain null values."
                    # Basic check for non-negativity for physical measurements
                    if col in ['duration_ms', 'avg_pitch_hz', 'max_energy']:
                        assert (df[col] >= 0).all(), f"Column '{col}' should contain non-negative values."

                for col in string_cols:
                    assert pd.api.types.is_string_dtype(df[col]), f"Column '{col}' should be string type."
                    assert not df[col].isnull().any(), f"Column '{col}' should not contain null values."

                assert pd.api.types.is_bool_dtype(df['is_vowel']), "Column 'is_vowel' should be boolean."
                assert not df['is_vowel'].isnull().any(), "Column 'is_vowel' should not contain null values."
                
                # Check phoneme_id uniqueness
                assert df['phoneme_id'].nunique() == expected_rows, "phoneme_id should be unique for each sample."
                assert pd.api.types.is_integer_dtype(df['phoneme_id']), "Column 'phoneme_id' should be integer."