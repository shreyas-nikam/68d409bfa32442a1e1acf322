import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Keep the definition_f15808e551db4573ab960c4f928c1563 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_f15808e551db4573ab960c4f928c1563 import train_simple_regression_model

@pytest.mark.parametrize("X_train_input, y_train_input, expected_outcome", [
    # Test Case 1: Successful training with valid numpy arrays.
    # Expects a fitted LinearRegression model.
    (np.array([[1], [2], [3], [4]]), np.array([2, 4, 5, 7]), LinearRegression),
    
    # Test Case 2: Successful training with valid pandas DataFrame and Series.
    # Expects a fitted LinearRegression model.
    (pd.DataFrame({'feature_a': [1, 2, 3, 4], 'feature_b': [5, 6, 7, 8]}), pd.Series([10, 12, 14, 16]), LinearRegression),
    
    # Test Case 3: Empty training data for both X_train and y_train.
    # Expects a ValueError as sklearn models require at least one sample.
    (np.array([]).reshape(0, 1), np.array([]), ValueError),
    
    # Test Case 4: Mismatched data lengths for X_train and y_train.
    # Expects a ValueError as features and target must have the same number of samples.
    (np.array([[1], [2], [3]]), np.array([1, 2]), ValueError),
    
    # Test Case 5: X_train contains non-numeric data (e.g., a categorical string column).
    # Expects a ValueError from scikit-learn when it attempts to convert non-numeric types.
    (pd.DataFrame({'numeric_f': [1, 2, 3], 'categorical_f': ['A', 'B', 'C']}), pd.Series([10, 20, 30]), ValueError),
])
def test_train_simple_regression_model(X_train_input, y_train_input, expected_outcome):
    """
    Tests the train_simple_regression_model function across various scenarios,
    including successful training and different types of invalid inputs.
    """
    try:
        model = train_simple_regression_model(X_train_input, y_train_input)
        
        # Assertions for successful training cases
        assert isinstance(model, expected_outcome)
        assert hasattr(model, 'coef_') # Check if model is fitted (coefficients exist)
        assert model.coef_ is not None # Coefficients should not be None
        assert hasattr(model, 'intercept_') # Check if model is fitted (intercept exists)
        assert model.intercept_ is not None # Intercept should not be None
        
        # Additional check for the number of coefficients matching features, for non-empty, 2D inputs
        if (isinstance(X_train_input, (pd.DataFrame, np.ndarray)) and 
            X_train_input.shape[0] > 0 and X_train_input.ndim > 1):
            assert len(model.coef_) == X_train_input.shape[1]
            
    except Exception as e:
        # Assertions for error cases: ensure the raised exception is of the expected type
        assert isinstance(e, expected_outcome)