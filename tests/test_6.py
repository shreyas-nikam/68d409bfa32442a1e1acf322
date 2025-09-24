import pytest
from unittest.mock import MagicMock
# This block must remain as is.
from definition_c5359519923140609c458fd4bf40fe59 import plot_loss_curves
import matplotlib.pyplot as plt

# Fixture to mock matplotlib.pyplot functions for all tests
# This prevents actual plotting windows from opening during tests
# and allows us to check if the plotting functions were called.
@pytest.fixture(autouse=True)
def mock_matplotlib(mocker):
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.plot')
    mocker.patch('matplotlib.pyplot.title')
    mocker.patch('matplotlib.pyplot.xlabel')
    mocker.patch('matplotlib.pyplot.ylabel')
    mocker.patch('matplotlib.pyplot.legend')
    mocker.patch('matplotlib.pyplot.grid')
    mocker.patch('matplotlib.pyplot.show')

# Helper function to create a mock Keras History object for testing
def create_mock_keras_history(loss_data, val_loss_data=None):
    mock_obj = MagicMock()
    mock_history_dict = {'loss': loss_data}
    if val_loss_data is not None:
        mock_history_dict['val_loss'] = val_loss_data
    mock_obj.history = mock_history_dict
    return mock_obj

@pytest.mark.parametrize(
    "history_input, expected_plot_calls, expected_legend_calls, expected_exception, expected_train_loss, expected_val_loss",
    [
        # Test Case 1: Valid dictionary history with both training and validation loss
        # Expected: two plot calls (for train and val loss), legend, titles, and show call.
        ({'loss': [0.5, 0.4, 0.3], 'val_loss': [0.6, 0.55, 0.45]}, 2, 1, None, [0.5, 0.4, 0.3], [0.6, 0.55, 0.45]),

        # Test Case 2: Valid dictionary history with only training loss
        # Expected: one plot call (for train loss), legend, titles, and show call.
        ({'loss': [0.5, 0.4, 0.3]}, 1, 1, None, [0.5, 0.4, 0.3], None),

        # Test Case 3: Empty dictionary history (edge case: no loss data)
        # Expected: no plot calls, no legend, but titles and show call (empty plot).
        ({}, 0, 0, None, None, None),

        # Test Case 4: Invalid input type (None) - edge case
        # Expected: an AttributeError (e.g., trying to access .get() or .history on None), no matplotlib calls.
        (None, 0, 0, AttributeError, None, None),

        # Test Case 5: Mocked TensorFlow Keras History object (expected functionality)
        # Assumes plot_loss_curves extracts history data from obj.history attribute.
        # Expected: two plot calls, legend, titles, and show call.
        (create_mock_keras_history([0.1, 0.08, 0.05], [0.12, 0.11, 0.09]), 2, 1, None, [0.1, 0.08, 0.05], [0.12, 0.11, 0.09]),
    ]
)
def test_plot_loss_curves(history_input, expected_plot_calls, expected_legend_calls, expected_exception, expected_train_loss, expected_val_loss):
    if expected_exception:
        # If an exception is expected, assert that it is raised
        with pytest.raises(expected_exception):
            plot_loss_curves(history_input)
        # Ensure no matplotlib calls were made if an exception occurred early
        plt.plot.assert_not_called()
        plt.show.assert_not_called()
        plt.title.assert_not_called()
        plt.xlabel.assert_not_called()
        plt.ylabel.assert_not_called()
        plt.legend.assert_not_called()
    else:
        # If no exception is expected, call the function
        plot_loss_curves(history_input)
        
        # Assert the total number of plt.plot calls
        assert plt.plot.call_count == expected_plot_calls
        
        # Check for specific plot calls if data was expected
        if expected_train_loss:
            plt.plot.assert_any_call(expected_train_loss, label='Training Loss')
        if expected_val_loss:
            plt.plot.assert_any_call(expected_val_loss, label='Validation Loss')

        # plt.show should always be called if the function runs without error
        plt.show.assert_called_once()
        
        # Check the number of plt.legend calls
        assert plt.legend.call_count == expected_legend_calls
        
        # plt.title, xlabel, ylabel should always be called if the function runs without error
        plt.title.assert_called_once()
        plt.xlabel.assert_called_once()
        plt.ylabel.assert_called_once()