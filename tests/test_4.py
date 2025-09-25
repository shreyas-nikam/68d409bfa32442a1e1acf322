import pytest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch, MagicMock
import os

# DO NOT REPLACE or REMOVE this block
from definition_fe7d670ede7049f383b5950bd103a583 import plot_categorical_bar_comparison
# DO NOT REPLACE or REMOVE this block

# Fixtures for different DataFrame scenarios
@pytest.fixture
def sample_df():
    """Returns a valid DataFrame for plotting."""
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A'],
        'value': [10, 20, 15, 25, 30, 35, 12],
        'other_col': ['x', 'y', 'z', 'x', 'y', 'z', 'x']
    })

@pytest.fixture
def empty_df():
    """Returns an empty DataFrame with expected columns."""
    return pd.DataFrame(columns=['category', 'value'], dtype=object)

@pytest.fixture
def non_numeric_value_df():
    """Returns a DataFrame where the value_column is non-numeric."""
    return pd.DataFrame({
        'category': ['A', 'B', 'A'],
        'value': ['ten', 'twenty', 'fifteen'] # Strings in value column
    })

@pytest.mark.parametrize(
    "df_input_type, category_col, value_col, title, x_label, y_label, expected_exception, df_data_key",
    [
        # Test Case 1: Happy Path - Valid Data
        ("fixture", "category", "value", "Test Plot Valid Data", "Category", "Value", None, "sample_df"),
        # Test Case 2: Empty DataFrame
        ("fixture", "category", "value", "Test Plot Empty DF", "Category", "Value", None, "empty_df"),
        # Test Case 3: Missing value_column
        ("fixture", "category", "non_existent_value", "Test Plot Missing Value Col", "Category", "Value", KeyError, "sample_df"),
        # Test Case 4: Non-numeric value_column
        ("fixture", "category", "value", "Test Plot Non Numeric Value", "Category", "Value", TypeError, "non_numeric_value_df"),
        # Test Case 5: Invalid df type (e.g., list instead of DataFrame)
        ("direct", "category", "value", "Test Plot Invalid DF Type", "Category", "Value", AttributeError, [1, 2, 3]),
    ]
)
def test_plot_categorical_bar_comparison(
    request,
    df_input_type,
    category_col,
    value_col,
    title,
    x_label,
    y_label,
    expected_exception,
    df_data_key # Used to get fixture value or direct data
):
    """
    Tests the plot_categorical_bar_comparison function for various scenarios.
    Mocks seaborn and matplotlib calls to prevent actual plotting and file I/O.
    """
    
    # Get the DataFrame input based on the parameterization
    if df_input_type == "fixture":
        df = request.getfixturevalue(df_data_key)
    else: # df_input_type == "direct"
        df = df_data_key # Pass the list directly
        
    plot_file_name = f"{title.replace(' ', '_').lower()}.png"

    # Patch seaborn.barplot and matplotlib functions
    with patch('seaborn.barplot') as mock_barplot, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.show') as mock_show, \
         patch('matplotlib.pyplot.clf') as mock_clf, \
         patch('matplotlib.pyplot.figure') as mock_figure: # Mock figure creation to control axes/title setting
        
        # Configure mock_figure to return a mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        if expected_exception:
            with pytest.raises(expected_exception):
                plot_categorical_bar_comparison(df, category_col, value_col, title, x_label, y_label)
            
            # On error, ensure no plotting or saving attempts were made
            mock_barplot.assert_not_called()
            mock_savefig.assert_not_called()
            mock_show.assert_not_called()
            mock_clf.assert_not_called()
        else:
            # Call the function for valid scenarios
            plot_categorical_bar_comparison(df, category_col, value_col, title, x_label, y_label)
            
            # Assertions for successful plot generation and saving
            mock_barplot.assert_called_once()
            mock_savefig.assert_called_once_with(plot_file_name)
            mock_show.assert_called_once()
            mock_clf.assert_called_once() # Good practice to clear the current figure

            # Further checks on seaborn.barplot call arguments (simplified)
            args, kwargs = mock_barplot.call_args
            assert 'x' in kwargs
            assert 'y' in kwargs
            assert 'data' in kwargs
            assert isinstance(kwargs['data'], (pd.DataFrame, pd.Series))
            
            # Check for palette usage (as per notebook spec "color-blind-friendly palette")
            assert 'palette' in kwargs 

            # Check if title and labels were set on the axes
            mock_ax.set_title.assert_called_once_with(title)
            mock_ax.set_xlabel.assert_called_once_with(x_label)
            mock_ax.set_ylabel.assert_called_once_with(y_label)