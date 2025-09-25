import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# BLOCK START
from definition_e3a6e818b8a44e2e846ef8e7038c5544 import create_simple_latent_features
# BLOCK END

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame for testing."""
    data = {
        'duration_ms': [100, 120, 90, 150, 110],
        'avg_pitch_hz': [150, 160, 140, 180, 155],
        'max_energy': [0.5, 0.6, 0.45, 0.7, 0.55],
        'phoneme_symbol': ['a', 'b', 'c', 'a', 'b'],
        'is_vowel': [True, False, True, True, False]
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_dataframe_with_cols():
    """Provides an empty DataFrame with defined columns."""
    return pd.DataFrame(columns=['duration_ms', 'avg_pitch_hz', 'max_energy'])

@pytest.mark.parametrize(
    "df_fixture, numeric_features_list, expected_shape, expected_exception",
    [
        # Test Case 1: Standard functionality - scale a subset of numeric features.
        # Should return a DataFrame with the specified columns scaled.
        ("sample_dataframe", ['duration_ms', 'avg_pitch_hz'], (5, 2), None),
        
        # Test Case 2: Scale all available numeric features.
        # Should return a DataFrame with all numeric columns scaled.
        ("sample_dataframe", ['duration_ms', 'avg_pitch_hz', 'max_energy'], (5, 3), None),
        
        # Test Case 3: Empty numeric_features_list.
        # Should return a DataFrame with the same number of rows but 0 columns.
        ("sample_dataframe", [], (5, 0), None),
        
        # Test Case 4: Non-existent column in numeric_features_list.
        # Accessing a non-existent column should raise a KeyError.
        ("sample_dataframe", ['duration_ms', 'non_existent_feature'], None, KeyError),
        
        # Test Case 5: Empty input DataFrame.
        # Should return an empty DataFrame with 0 rows and columns matching the list.
        ("empty_dataframe_with_cols", ['duration_ms', 'avg_pitch_hz'], (0, 2), None),
    ]
)
def test_create_simple_latent_features(
    request, df_fixture, numeric_features_list, expected_shape, expected_exception
):
    df = request.getfixturevalue(df_fixture)

    if expected_exception:
        with pytest.raises(expected_exception):
            create_simple_latent_features(df, numeric_features_list)
    else:
        result_df = create_simple_latent_features(df, numeric_features_list)

        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape == expected_shape
        
        if expected_shape[0] > 0 and expected_shape[1] > 0:
            # Verify columns are correct
            assert list(result_df.columns) == numeric_features_list
            
            # For StandardScaler, mean should be close to 0 and std dev close to 1
            # Using atol for floating point comparisons
            for col in result_df.columns:
                assert np.isclose(result_df[col].mean(), 0.0, atol=1e-9)
                assert np.isclose(result_df[col].std(), 1.0, atol=1e-9)
        elif expected_shape[1] == 0 and expected_shape[0] > 0:
            # Case for empty numeric_features_list with non-empty input df
            assert result_df.columns.empty
            assert result_df.shape[0] == df.shape[0]
        elif expected_shape[0] == 0:
            # Case for empty input df
            assert result_df.shape[0] == 0
            assert list(result_df.columns) == numeric_features_list