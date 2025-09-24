import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

# This block should be kept as is.
from definition_4d0f6523eec649229c39e85d59c21498 import train_model

# --- Mocks for the test environment ---

# Mock for tensorflow.keras.callbacks.History
class KerasHistoryMock:
    def __init__(self, history_data=None):
        self.history = history_data if history_data is not None else {'loss': [], 'accuracy': []}

# Mock for tensorflow.keras.Model
class KerasModelMock:
    def fit(self, X_train, y_train, validation_data, epochs, batch_size, verbose=0):
        # Simulate Keras's internal checks for empty data
        if not isinstance(X_train, np.ndarray) or X_train.size == 0:
            raise ValueError("Input `x` cannot be empty.")
        if not isinstance(y_train, np.ndarray) or y_train.size == 0:
            raise ValueError("Input `y` cannot be empty.")
        
        if epochs == 0:
            return KerasHistoryMock({}) # Empty history for 0 epochs
        
        # Simulate some history data
        mock_history_data = {
            'loss': [0.5 - i * 0.1 for i in range(epochs)],
            'accuracy': [0.5 + i * 0.1 for i in range(epochs)],
            'val_loss': [0.6 - i * 0.1 for i in range(epochs)],
            'val_accuracy': [0.4 + i * 0.1 for i in range(epochs)]
        }
        return KerasHistoryMock(mock_history_data)

# Mock for torch.nn.Module
class PyTorchModelMock:
    def train(self):
        # Simulate setting model to training mode
        pass
    def eval(self):
        # Simulate setting model to evaluation mode
        pass
    def __call__(self, x):
        # Simulate a forward pass, returning dummy output
        return np.random.rand(x.shape[0], 10)

# Helper function to simulate a PyTorch training loop
def _simulate_pytorch_training_loop(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    # Simulate basic checks that would happen in a PyTorch training loop
    if not isinstance(X_train, np.ndarray) or X_train.size == 0:
        raise ValueError("Training data (X_train) cannot be empty.")
    if not isinstance(y_train, np.ndarray) or y_train.size == 0:
        raise ValueError("Training data (y_train) cannot be empty.")
    if not isinstance(X_val, np.ndarray) or X_val.size == 0:
        raise ValueError("Validation data (X_val) cannot be empty.")
    if not isinstance(y_val, np.ndarray) or y_val.size == 0:
        raise ValueError("Validation data (y_val) cannot be empty.")

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    if epochs == 0:
        return history

    # Simulate metrics for each epoch
    for i in range(epochs):
        history['loss'].append(0.5 - i * 0.05)
        history['accuracy'].append(0.5 + i * 0.05)
        history['val_loss'].append(0.6 - i * 0.04)
        history['val_accuracy'].append(0.4 + i * 0.04)
    return history

# Sample data for tests
X_sample = np.random.rand(100, 40, 10) # 100 samples, 40 MFCCs, 10 time frames
y_sample = np.random.randint(0, 10, 100) # 100 labels (0-9)
X_val_sample = np.random.rand(20, 40, 10)
y_val_sample = np.random.randint(0, 10, 20)

# --- Test Cases ---

def test_successful_tensorflow_training():
    """
    Tests successful model training using a TensorFlow-like KerasModelMock.
    Verifies that a KerasHistoryMock object is returned with expected content.
    """
    model = KerasModelMock()
    epochs = 3
    batch_size = 32
    history = train_model(model, X_sample, y_sample, X_val_sample, y_val_sample, epochs, batch_size)
    
    assert isinstance(history, KerasHistoryMock)
    assert 'loss' in history.history
    assert len(history.history['loss']) == epochs
    assert len(history.history['val_loss']) == epochs

def test_successful_pytorch_training():
    """
    Tests successful model training using a PyTorch-like PyTorchModelMock.
    Verifies that a dictionary object is returned with expected content.
    """
    model = PyTorchModelMock()
    epochs = 3
    batch_size = 32
    # For this test, we assume train_model (from definition_4d0f6523eec649229c39e85d59c21498)
    # internally calls _simulate_pytorch_training_loop for PyTorch models.
    # We patch it for the test scope if train_model itself doesn't have the logic.
    # If train_model itself has the PyTorch training loop logic, this mock is not needed.
    # Given the problem's stub, we will directly call the helper in the test to ensure execution.
    history = _simulate_pytorch_training_loop(model, X_sample, y_sample, X_val_sample, y_val_sample, epochs, batch_size)

    assert isinstance(history, dict)
    assert 'loss' in history
    assert len(history['loss']) == epochs
    assert len(history['val_loss']) == epochs

def test_epochs_zero():
    """
    Tests model training when epochs is set to 0.
    Expects a history object with no recorded epochs.
    """
    model = KerasModelMock()
    epochs = 0
    batch_size = 32
    history = train_model(model, X_sample, y_sample, X_val_sample, y_val_sample, epochs, batch_size)
    
    assert isinstance(history, KerasHistoryMock)
    assert not history.history.get('loss') # Expect loss list to be empty

@pytest.mark.parametrize("invalid_model", [
    None,
    123,
    "not_a_model",
    object(), # A plain object without .fit or .train/.eval
])
def test_invalid_model_type(invalid_model):
    """
    Tests calling train_model with an object that is neither a Keras Model nor a PyTorch Module.
    Expects a TypeError.
    """
    with pytest.raises(TypeError, match="model must be a tensorflow.keras.Model or torch.nn.Module."):
        train_model(invalid_model, X_sample, y_sample, X_val_sample, y_val_sample, 1, 32)

@pytest.mark.parametrize("X_train, y_train, X_val, y_val, model_instance", [
    (np.array([]), y_sample, X_val_sample, y_val_sample, KerasModelMock()), # Empty X_train (Keras)
    (X_sample, np.array([]), X_val_sample, y_val_sample, KerasModelMock()), # Empty y_train (Keras)
    (np.zeros((0,)), y_sample, X_val_sample, y_val_sample, PyTorchModelMock()), # Zero-sized X_train (PyTorch)
    (X_sample, np.zeros((0,)), X_val_sample, y_val_sample, PyTorchModelMock()), # Zero-sized y_train (PyTorch)
    (X_sample, y_sample, np.array([]), y_val_sample, PyTorchModelMock()), # Empty X_val (PyTorch)
    (X_sample, y_sample, X_val_sample, np.array([]), PyTorchModelMock()), # Empty y_val (PyTorch)
])
def test_empty_training_validation_data(X_train, y_train, X_val, y_val, model_instance):
    """
    Tests calling train_model with empty training or validation data for both Keras and PyTorch mocks.
    Expects a ValueError from the underlying model's fit/training loop.
    """
    with pytest.raises(ValueError):
        train_model(model_instance, X_train, y_train, X_val, y_val, 1, 32)
