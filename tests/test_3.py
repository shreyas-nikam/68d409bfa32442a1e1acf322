import pytest
import pandas as pd
from unittest.mock import MagicMock

# Keep the definition_4b086f52003e4aaf90af5ef472d9e562 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_4b086f52003e4aaf90af5ef472d9e562 import plot_scatter_relationship

@pytest.fixture
def sample_df():
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame({
        'duration_ms': [100, 120, 80, 150, 110],
        'avg_pitch_hz': [150, 200, 130, 220, 180],
        'max_energy': [0.5, 0.7, 0.4, 0.8, 0.6],
        'is_vowel': [True, False, True, False, True],
        'dialect': ['A', 'B', 'A', 'C', 'B']
    })

@pytest.fixture
def empty_df():
    """Provides an empty DataFrame for edge case testing."""
    return pd.DataFrame()

@pytest.mark.parametrize(
    "df_fixture_name, x_column, y_column, hue_column, title, x_label, y_label, expected_exception",
    [
        # Test Case 1: Basic functionality - numeric columns, no hue
        ('sample_df', 'duration_ms', 'avg_pitch_hz', None, "Duration vs Pitch", "Duration (ms)", "Pitch (Hz)", None),
        
        # Test Case 2: Functionality with a categorical hue column
        ('sample_df', 'max_energy', 'avg_pitch_hz', 'is_vowel', "Energy vs Pitch by Vowel", "Max Energy", "Pitch (Hz)", None),
        
        # Test Case 3: Non-existent x_column - should raise KeyError
        ('sample_df', 'non_existent_x', 'avg_pitch_hz', None, "Invalid X Column", "X", "Y", KeyError),
        
        # Test Case 4: Non-existent hue_column - should raise KeyError
        ('sample_df', 'duration_ms', 'avg_pitch_hz', 'non_existent_hue', "Invalid Hue Column", "X", "Y", KeyError),
        
        # Test Case 5: Empty DataFrame - accessing any column should raise KeyError
        ('empty_df', 'duration_ms', 'avg_pitch_hz', None, "Empty Dataframe Plot", "X", "Y", KeyError),
    ]
)
def test_plot_scatter_relationship(df_fixture_name, x_column, y_column, hue_column, title, x_label, y_label, expected_exception, request, mocker):
    """
    Tests the plot_scatter_relationship function for various scenarios including
    expected functionality and edge cases with invalid column names or empty data.
    """
    df = request.getfixturevalue(df_fixture_name)

    # Mock all external calls to matplotlib.pyplot and seaborn
    mock_scatterplot = mocker.patch('seaborn.scatterplot')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_show = mocker.patch('matplotlib.pyplot.show')
    mock_clf = mocker.patch('matplotlib.pyplot.clf')
    
    # Mock figure/subplots creation as seaborn might use it or the function might explicitly
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.subplots')
    
    # Mock axis label and title setting functions which would typically be called on an Axes object
    mock_ax = MagicMock()
    mock_scatterplot.return_value = mock_ax # Assume scatterplot returns an Axes object
    mocker.patch('matplotlib.pyplot.gca', return_value=mock_ax) # Mock get current axis for labels/title

    try:
        plot_scatter_relationship(df, x_column, y_column, hue_column, title, x_label, y_label)
        
        # If no exception was expected, assert that plotting functions were called correctly
        assert expected_exception is None, "An unexpected exception occurred."
        
        mock_scatterplot.assert_called_once_with(
            data=df, x=x_column, y=y_column, hue=hue_column
        )
        mock_savefig.assert_called_once_with(f'{title}.png')
        mock_show.assert_called_once()
        mock_clf.assert_called_once() # Assuming matplotlib cleanup after plot is saved/shown

        # Additionally, verify that labels and title were set on the axes
        mock_ax.set_title.assert_called_once_with(title)
        mock_ax.set_xlabel.assert_called_once_with(x_label)
        mock_ax.set_ylabel.assert_called_once_with(y_label)

    except Exception as e:
        # If an exception was expected, assert its type
        assert expected_exception is not None, f"Expected no exception, but got {type(e).__name__}"
        assert isinstance(e, expected_exception), f"Expected {expected_exception.__name__}, but got {type(e).__name__}"
        
        # Ensure that no plotting functions were called if an error occurred
        mock_scatterplot.assert_not_called()
        mock_savefig.assert_not_called()
        mock_show.assert_not_called()
        mock_clf.assert_not_called()
        mock_ax.set_title.assert_not_called()
        mock_ax.set_xlabel.assert_not_called()
        mock_ax.set_ylabel.assert_not_called()