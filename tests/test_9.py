import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

# definition_9ade430a9e274e959a9ea5a8302f8fbd block as per instruction
from definition_9ade430a9e274e959a9ea5a8302f8fbd import interactive_phoneme_analyzer

# Helper function to create a mock DataFrame for testing
def _create_mock_df(data=None, columns=None, is_empty=False):
    """Creates a mock pandas DataFrame with necessary methods for testing."""
    mock_df = MagicMock(spec=pd.DataFrame)
    
    if is_empty:
        mock_df.empty = True
        mock_df.columns = columns if columns else ['phoneme_symbol', 'duration_ms', 'avg_pitch_hz', 'max_energy', 'pronunciation_naturalness_score']
        mock_df.groupby.return_value.mean.return_value = pd.DataFrame(columns=mock_df.columns, index=pd.Index([], name='phoneme_symbol'))
        mock_df.mean.return_value = pd.Series({col: float('nan') for col in mock_df.columns if col not in ['phoneme_symbol']})
        mock_df.__getitem__.return_value = mock_df # Filtering an empty DF returns an empty DF
    else:
        if data is None:
            data = {
                'phoneme_symbol': ['a', 'b', 'a', 'b', 'c'],
                'duration_ms': [100, 120, 110, 130, 90],
                'avg_pitch_hz': [150, 160, 155, 165, 140],
                'max_energy': [0.5, 0.6, 0.55, 0.65, 0.45],
                'pronunciation_naturalness_score': [0.8, 0.7, 0.85, 0.75, 0.9]
            }
        if columns is None:
            columns = list(data.keys())
        
        mock_df.columns = columns
        mock_df.empty = False

        # Mocking groupby and mean operations
        mock_grouped_df = MagicMock()
        mock_grouped_df.mean.return_value = pd.DataFrame({
            'duration_ms': [105, 125, 90],
            'avg_pitch_hz': [152.5, 162.5, 140],
            'max_energy': [0.525, 0.625, 0.45],
            'pronunciation_naturalness_score': [0.825, 0.725, 0.9]
        }, index=pd.Index(['a', 'b', 'c'], name='phoneme_symbol'))
        mock_df.groupby.return_value = mock_grouped_df

        mock_df.mean.return_value = pd.Series({
            'duration_ms': 110,
            'avg_pitch_hz': 154,
            'max_energy': 0.53,
            'pronunciation_naturalness_score': 0.81
        })

        # Mocking filtering for a specific phoneme (e.g., df[df['phoneme_symbol'] == 'a'])
        mock_filtered_df = MagicMock(spec=pd.DataFrame)
        mock_filtered_df.mean.return_value = pd.Series({
            'duration_ms': 105,
            'avg_pitch_hz': 152.5,
            'max_energy': 0.525,
            'pronunciation_naturalness_score': 0.825
        })
        mock_df.__getitem__.return_value = mock_filtered_df
        
    return mock_df

# Patching ipywidgets and matplotlib globally for the test file to avoid boilerplate
@patch('ipywidgets.Dropdown', autospec=True)
@patch('ipywidgets.Output', autospec=True)
@patch('ipywidgets.VBox', autospec=True)
@patch('IPython.display.display', autospec=True)
@patch('matplotlib.pyplot.bar', autospec=True)
@patch('matplotlib.pyplot.savefig', autospec=True)
@patch('matplotlib.pyplot.show', autospec=True)
@patch('matplotlib.pyplot.close', autospec=True)
@patch('matplotlib.pyplot.figure', autospec=True)
@patch('matplotlib.pyplot.title', autospec=True)
@patch('matplotlib.pyplot.xlabel', autospec=True)
@patch('matplotlib.pyplot.ylabel', autospec=True)
class TestInteractivePhonemeAnalyzer:
    """
    Collection of tests for interactive_phoneme_analyzer.
    All tests inherit the common patches for ipywidgets and matplotlib.
    """

    def test_basic_functionality_widget_creation(self, 
        mock_ylabel, mock_xlabel, mock_title, mock_figure, mock_close, mock_show, mock_savefig, 
        mock_bar, mock_display, mock_vbox, mock_output, mock_dropdown
    ):
        """
        Test 1: Ensures the interactive widget components (Dropdown, Output, VBox)
        are created and displayed correctly with valid inputs. Does not simulate interaction.
        """
        mock_df = _create_mock_df()
        phoneme_symbols = ['a', 'b', 'c']

        interactive_phoneme_analyzer(mock_df, phoneme_symbols)

        # Assert that a Dropdown widget was created with the correct options
        mock_dropdown.assert_called_once_with(
            options=phoneme_symbols,
            description='Select Phoneme:',
            disabled=False,
        )

        # Assert that an Output widget was created
        mock_output.assert_called_once()

        # Assert that ipywidgets.VBox was used to arrange the widgets
        mock_vbox.assert_called_once_with((mock_dropdown.return_value, mock_output.return_value))

        # Assert that the assembled widget (VBox instance) was displayed
        mock_display.assert_called_once_with(mock_vbox.return_value)

        # Verify that the observe method was called on the dropdown instance to set up the callback
        dropdown_instance = mock_dropdown.return_value
        dropdown_instance.observe.assert_called_once()
        assert callable(dropdown_instance.observe.call_args[0][0]) # Check that a function was passed
        assert dropdown_instance.observe.call_args[1]['names'] == 'value'

        # Verify initial rendering into the output widget by checking its context manager
        mock_output.return_value.__enter__.assert_called_once()
        mock_output.return_value.__exit__.assert_called_once()

    def test_empty_phoneme_symbols_list(self,
        mock_ylabel, mock_xlabel, mock_title, mock_figure, mock_close, mock_show, mock_savefig, 
        mock_bar, mock_display, mock_vbox, mock_output, mock_dropdown
    ):
        """
        Test 2: Handles an empty list of phoneme symbols gracefully.
        The dropdown should be created with empty options, and no crash should occur.
        """
        mock_df = _create_mock_df()
        phoneme_symbols = []

        interactive_phoneme_analyzer(mock_df, phoneme_symbols)

        mock_dropdown.assert_called_once_with(
            options=[],
            description='Select Phoneme:',
            disabled=False, # Assuming it's not explicitly disabled, but just empty
        )
        mock_display.assert_called_once()
        mock_output.assert_called_once()
        mock_vbox.assert_called_once_with((mock_dropdown.return_value, mock_output.return_value))
        
        dropdown_instance = mock_dropdown.return_value
        # If the list is empty, there's no initial selection, so `observe` might not be triggered
        # or it might be set up but the callback for initial value will handle empty data.
        # It's reasonable that `observe` is still set up to handle future potential options (if dropdown was dynamic)
        dropdown_instance.observe.assert_called_once() # The callback mechanism is established
        # The output should still be cleared/used, even if it just says "No phonemes available"
        mock_output.return_value.__enter__.assert_called_once()
        mock_output.return_value.__exit__.assert_called_once()


    def test_df_with_missing_critical_columns(self,
        mock_ylabel, mock_xlabel, mock_title, mock_figure, mock_close, mock_show, mock_savefig, 
        mock_bar, mock_display, mock_vbox, mock_output, mock_dropdown
    ):
        """
        Test 3: Ensures a KeyError is raised when the DataFrame is missing critical columns.
        This error should occur during the initial data processing (e.g., calculating overall averages).
        """
        # Create a mock df missing 'duration_ms' and others that are expected
        mock_df_bad = _create_mock_df(columns=['phoneme_symbol', 'unrelated_col'])
        # Configure the mock to raise KeyError when expected columns are accessed
        # This simulates `df['duration_ms']` failing.
        def mock_getitem(key):
            if isinstance(key, list) and 'duration_ms' in key:
                raise KeyError("Column 'duration_ms' not found")
            if key == 'phoneme_symbol':
                # Return a mock series for phoneme_symbol so the dropdown options can be built
                mock_series = MagicMock(spec=pd.Series)
                mock_series.unique.return_value = ['p', 'q']
                return mock_series
            raise KeyError(f"Column '{key}' not found in mock_df_bad")

        mock_df_bad.__getitem__.side_effect = mock_getitem
        mock_df_bad.groupby.side_effect = KeyError("Column 'duration_ms' not found during grouping")
        mock_df_bad.mean.side_effect = KeyError("Column 'duration_ms' not found during overall mean calculation")


        phoneme_symbols = ['p', 'q'] # Even if valid, the df itself is bad

        with pytest.raises(KeyError) as excinfo:
            interactive_phoneme_analyzer(mock_df_bad, phoneme_symbols)
        
        # Check for error message indicating a missing column
        assert "Column 'duration_ms' not found" in str(excinfo.value) or "missing a required characteristic" in str(excinfo.value) # More general check


    def test_invalid_df_type(self,
        mock_ylabel, mock_xlabel, mock_title, mock_figure, mock_close, mock_show, mock_savefig, 
        mock_bar, mock_display, mock_vbox, mock_output, mock_dropdown
    ):
        """
        Test 4: Ensures an appropriate error (TypeError or AttributeError) is raised
        when `df` is not a pandas DataFrame.
        """
        invalid_df = "not a dataframe"
        phoneme_symbols = ['a', 'b', 'c']

        with pytest.raises((TypeError, AttributeError)) as excinfo:
            interactive_phoneme_analyzer(invalid_df, phoneme_symbols)
        
        # Check for typical error messages from pandas when methods are called on wrong type
        assert "has no attribute 'groupby'" in str(excinfo.value) or "expected a pandas DataFrame" in str(excinfo.value).lower()


    def test_empty_dataframe_with_valid_phoneme_symbols(self,
        mock_ylabel, mock_xlabel, mock_title, mock_figure, mock_close, mock_show, mock_savefig, 
        mock_bar, mock_display, mock_vbox, mock_output, mock_dropdown
    ):
        """
        Test 5: Handles an empty pandas DataFrame with valid phoneme symbols.
        The function should display zeros or NaNs for averages and not crash.
        """
        mock_df_empty = _create_mock_df(is_empty=True)
        phoneme_symbols = ['a', 'b']

        interactive_phoneme_analyzer(mock_df_empty, phoneme_symbols)

        mock_dropdown.assert_called_once_with(
            options=phoneme_symbols,
            description='Select Phoneme:',
            disabled=False,
        )
        mock_display.assert_called_once()
        mock_output.assert_called_once()
        mock_vbox.assert_called_once_with((mock_dropdown.return_value, mock_output.return_value))
        
        dropdown_instance = mock_dropdown.return_value
        dropdown_instance.observe.assert_called_once()

        mock_output.return_value.__enter__.assert_called_once()
        mock_output.return_value.__exit__.assert_called_once()

        # Depending on implementation, `matplotlib.pyplot.bar` or `savefig` might still be called
        # but with NaN values or an empty plot. Asserting no crash is the primary goal.
        # No specific assertions on plot content here without direct callback invocation.