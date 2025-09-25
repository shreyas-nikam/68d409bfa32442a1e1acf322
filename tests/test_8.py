import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import io
import sys

# Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_89ba5934421d4c12b45561341a679b52 import evaluate_regression_model

# Helper fixture to capture stdout for testing print statements
@pytest.fixture
def capsys_output(capsys):
    """Fixture to capture stdout and stderr."""
    return capsys

# Test Case 1: Standard Evaluation with valid numpy array inputs
def test_evaluate_regression_model_standard_numpy_input(mocker, capsys_output):
    """
    Tests the standard functionality where a model is evaluated on valid numpy arrays,
    and the R-squared and MAE are correctly calculated and printed.
    """
    # Mock the model and its predict method
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

    # Create dummy test data using numpy arrays
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Mock sklearn.metrics functions to return known values
    mocker.patch('sklearn.metrics.r2_score', return_value=0.95)
    mocker.patch('sklearn.metrics.mean_absolute_error', return_value=0.1)

    # Call the function to be tested
    evaluate_regression_model(mock_model, X_test, y_test)

    # Capture printed output
    captured = capsys_output.readouterr()

    # Assert that the output contains the expected metrics
    assert "R-squared Score: 0.95" in captured.out
    assert "Mean Absolute Error (MAE): 0.1" in captured.out
    
    # Assert that the model's predict method was called correctly
    mock_model.predict.assert_called_once_with(X_test)
    
    # Assert that metrics functions were called with the correct arguments (true values and predicted values)
    mocker.patch('sklearn.metrics.r2_score').assert_called_once_with(y_test, mock_model.predict.return_value)
    mocker.patch('sklearn.metrics.mean_absolute_error').assert_called_once_with(y_test, mock_model.predict.return_value)

# Test Case 2: Empty Test Data
def test_evaluate_regression_model_empty_data(mocker):
    """
    Tests the behavior when X_test and y_test are empty.
    Expects a ValueError from scikit-learn metrics functions due to empty inputs.
    """
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array([]) # Model predicts an empty array

    X_test = pd.DataFrame([]) # Empty DataFrame
    y_test = pd.Series([])    # Empty Series

    # Mock sklearn.metrics functions to raise a ValueError for empty inputs
    mocker.patch('sklearn.metrics.r2_score', side_effect=ValueError("y_true and y_pred must be non-empty."))
    mocker.patch('sklearn.metrics.mean_absolute_error', side_effect=ValueError("y_true and y_pred must be non-empty."))

    with pytest.raises(ValueError, match="y_true and y_pred must be non-empty."):
        evaluate_regression_model(mock_model, X_test, y_test)

    mock_model.predict.assert_called_once_with(X_test)
    # Ensure that the metrics functions were attempted to be called
    assert mocker.patch('sklearn.metrics.r2_score').called
    assert mocker.patch('sklearn.metrics.mean_absolute_error').called

# Test Case 3: y_test contains non-numeric values
def test_evaluate_regression_model_non_numeric_y_test(mocker):
    """
    Tests with non-numeric values in y_test.
    Expects a TypeError or ValueError from scikit-learn metrics functions.
    """
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array([1.1, 2.2, 3.3])

    X_test = pd.DataFrame(np.random.rand(3, 2))
    # y_test with a non-numeric value ('invalid')
    y_test = pd.Series([1.0, 'invalid', 3.0])

    # Sklearn metrics functions would typically raise a TypeError or ValueError for non-numeric data
    mocker.patch('sklearn.metrics.r2_score', side_effect=TypeError("Non-numeric data in y_true."))
    mocker.patch('sklearn.metrics.mean_absolute_error', side_effect=TypeError("Non-numeric data in y_true."))

    with pytest.raises(TypeError, match="Non-numeric data in y_true."):
        evaluate_regression_model(mock_model, X_test, y_test)

    mock_model.predict.assert_called_once_with(X_test)
    assert mocker.patch('sklearn.metrics.r2_score').called
    assert mocker.patch('sklearn.metrics.mean_absolute_error').called

# Test Case 4: Model object is missing the 'predict' method
def test_evaluate_regression_model_no_predict_method():
    """
    Tests with a model object that does not have a 'predict' method.
    Expects an AttributeError.
    """
    # Create a dummy object that simulates a model but lacks a 'predict' method
    class BadModel:
        pass
    
    mock_model = BadModel()
    X_test = pd.DataFrame(np.random.rand(5, 2))
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    # Expect an AttributeError when the function tries to call model.predict()
    with pytest.raises(AttributeError):
        evaluate_regression_model(mock_model, X_test, y_test)

# Test Case 5: Mismatched shapes of y_test and y_pred
def test_evaluate_regression_model_mismatched_y_shapes(mocker):
    """
    Tests when the true target values (y_test) and predicted values (y_pred)
    have a different number of samples.
    Expects a ValueError from scikit-learn metrics functions.
    """
    mock_model = mocker.MagicMock()
    # Model predicts 5 values based on X_test, but y_test only has 3 values
    mock_model.predict.return_value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

    X_test = pd.DataFrame(np.random.rand(5, 2)) # X_test has 5 samples
    y_test = pd.Series([1.0, 2.0, 3.0])         # y_test has 3 samples

    # Sklearn metrics will raise ValueError for mismatched input shapes/lengths
    mocker.patch('sklearn.metrics.r2_score', side_effect=ValueError("y_true and y_pred must have same length."))
    mocker.patch('sklearn.metrics.mean_absolute_error', side_effect=ValueError("y_true and y_pred must have same length."))

    with pytest.raises(ValueError, match="y_true and y_pred must have same length."):
        evaluate_regression_model(mock_model, X_test, y_test)

    mock_model.predict.assert_called_once_with(X_test)
    assert mocker.patch('sklearn.metrics.r2_score').called
    assert mocker.patch('sklearn.metrics.mean_absolute_error').called