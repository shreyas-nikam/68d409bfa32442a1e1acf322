import pytest
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# definition_737950724b8040cfa00eb2f559ef5aba block
# DO NOT REPLACE or REMOVE this block
from definition_737950724b8040cfa00eb2f559ef5aba import plot_distribution_histogram
# END definition_737950724b8040cfa00eb2f559ef5aba block


# Fixture to create a dummy DataFrame for testing
@pytest.fixture
def sample_dataframe():
    """Provides a sample pandas DataFrame with numeric, string, and boolean columns."""
    data = {
        'duration_ms': np.random.normal(loc=100, scale=20, size=100).astype(int),
        'avg_pitch_hz': np.random.normal(loc=150, scale=30, size=100).astype(int),
        'category_col': [f'cat_{i%3}' for i in range(100)],
        'bool_col': [i%2 == 0 for i in range(100)]
    }
    return pd.DataFrame(data)

@patch('seaborn.histplot')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure') # Mock figure creation
@patch('matplotlib.pyplot.clf')    # Mock figure clearing
@patch('matplotlib.pyplot.close')  # Mock figure closing
def test_plot_distribution_histogram_happy_path_default_bins(
    mock_close, mock_clf, mock_figure, mock_savefig, mock_ylabel, mock_xlabel, mock_title, mock_histplot, sample_dataframe
):
    """
    Test Case 1: Verifies the function correctly generates a histogram for a valid numeric column
    using default bins, and saves the plot.
    """
    df = sample_dataframe
    column = 'duration_ms'
    title = 'Duration Distribution'
    x_label = 'Duration (ms)'
    y_label = 'Frequency'
    expected_filename = f'{column}_distribution_histogram.png'

    plot_distribution_histogram(df, column, title, x_label, y_label, bins=30) # Call with explicit default

    mock_histplot.assert_called_once_with(data=df, x=column, bins=30)
    mock_title.assert_called_once_with(title)
    mock_xlabel.assert_called_once_with(x_label)
    mock_ylabel.assert_called_once_with(y_label)
    mock_savefig.assert_called_once_with(expected_filename)
    mock_figure.assert_called_once() # Ensure a figure was created
    mock_clf.assert_called_once()    # Ensure figure was cleared
    mock_close.assert_called_once()  # Ensure figure was closed

@patch('seaborn.histplot')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.clf')
@patch('matplotlib.pyplot.close')
def test_plot_distribution_histogram_custom_bins(
    mock_close, mock_clf, mock_figure, mock_savefig, mock_ylabel, mock_xlabel, mock_title, mock_histplot, sample_dataframe
):
    """
    Test Case 2: Checks if the function handles a custom number of bins correctly.
    """
    df = sample_dataframe
    column = 'avg_pitch_hz'
    title = 'Pitch Distribution'
    x_label = 'Pitch (Hz)'
    y_label = 'Count'
    custom_bins = 15
    expected_filename = f'{column}_distribution_histogram.png'

    plot_distribution_histogram(df, column, title, x_label, y_label, bins=custom_bins)

    mock_histplot.assert_called_once_with(data=df, x=column, bins=custom_bins)
    mock_title.assert_called_once_with(title)
    mock_xlabel.assert_called_once_with(x_label)
    mock_ylabel.assert_called_once_with(y_label)
    mock_savefig.assert_called_once_with(expected_filename)
    mock_figure.assert_called_once()
    mock_clf.assert_called_once()
    mock_close.assert_called_once()

@patch('seaborn.histplot')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.clf')
@patch('matplotlib.pyplot.close')
def test_plot_distribution_histogram_non_existent_column(
    mock_close, mock_clf, mock_figure, mock_savefig, mock_ylabel, mock_xlabel, mock_title, mock_histplot, sample_dataframe
):
    """
    Test Case 3: Ensures a KeyError is raised when the specified column does not exist in the DataFrame.
    """
    df = sample_dataframe
    column = 'non_existent_column'
    title = 'Non-existent Column'
    x_label = 'X'
    y_label = 'Y'

    # Mock histplot to simulate the KeyError that would occur when pandas tries to access a non-existent column
    mock_histplot.side_effect = KeyError(f"'{column}' not in index")

    with pytest.raises(KeyError, match=f"'{column}' not in index"):
        plot_distribution_histogram(df, column, title, x_label, y_label, bins=30)

    mock_histplot.assert_called_once() # The plot function would try to access the column, leading to error
    mock_savefig.assert_not_called()
    mock_title.assert_not_called()
    mock_xlabel.assert_not_called()
    mock_ylabel.assert_not_called()
    mock_figure.assert_called_once() # A figure might be created before the error
    mock_close.assert_called_once() # The figure should still be closed

@patch('seaborn.histplot')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.clf')
@patch('matplotlib.pyplot.close')
def test_plot_distribution_histogram_non_numeric_column(
    mock_close, mock_clf, mock_figure, mock_savefig, mock_ylabel, mock_xlabel, mock_title, mock_histplot, sample_dataframe
):
    """
    Test Case 4: Verifies that an appropriate error (e.g., TypeError or ValueError) is raised
    when trying to plot a non-numeric column as a histogram.
    """
    df = sample_dataframe
    column = 'category_col' # A string column
    title = 'Categorical Column'
    x_label = 'Categories'
    y_label = 'Count'

    # Mock histplot to raise the kind of error seaborn would for non-numeric data
    mock_histplot.side_effect = TypeError("Can't plot non-numeric data for histogram 'x'")

    with pytest.raises(TypeError, match="Can't plot non-numeric data for histogram 'x'"):
        plot_distribution_histogram(df, column, title, x_label, y_label, bins=30)

    mock_histplot.assert_called_once()
    mock_savefig.assert_not_called()
    mock_title.assert_not_called()
    mock_xlabel.assert_not_called()
    mock_ylabel.assert_not_called()
    mock_figure.assert_called_once()
    mock_close.assert_called_once()

@patch('seaborn.histplot')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.clf')
@patch('matplotlib.pyplot.close')
def test_plot_distribution_histogram_empty_dataframe(
    mock_close, mock_clf, mock_figure, mock_savefig, mock_ylabel, mock_xlabel, mock_title, mock_histplot
):
    """
    Test Case 5: Ensures proper error handling or behavior with an empty DataFrame.
    """
    df = pd.DataFrame({'numeric_col': []}) # DataFrame with an empty numeric column
    column = 'numeric_col'
    title = 'Empty Data Distribution'
    x_label = 'Value'
    y_label = 'Frequency'

    # Mock histplot to raise an error if called with empty data (common behavior)
    mock_histplot.side_effect = ValueError("Input data must not be empty for plotting.")

    with pytest.raises(ValueError, match="Input data must not be empty for plotting."):
        plot_distribution_histogram(df, column, title, x_label, y_label, bins=30)

    mock_histplot.assert_called_once()
    mock_savefig.assert_not_called()
    mock_title.assert_not_called()
    mock_xlabel.assert_not_called()
    mock_ylabel.assert_not_called()
    mock_figure.assert_called_once()
    mock_close.assert_called_once()