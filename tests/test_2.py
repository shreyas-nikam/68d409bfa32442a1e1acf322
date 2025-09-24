import pytest
from definition_96413a9560cc4a5187fed4ec27eef824 import create_model
import numpy as np

def test_create_model_valid_input_shape_and_num_labels():
    """Test that the function returns a model when given a valid input shape and number of labels."""
    input_shape = (40, 100, 1)
    num_labels = 10
    model = create_model(input_shape, num_labels)
    assert model is not None, "Model creation failed"

def test_create_model_with_small_input_shape():
    """Test the function with a smaller input shape."""
    input_shape = (10, 10, 1)
    num_labels = 5
    model = create_model(input_shape, num_labels)
    assert model is not None, "Model creation failed with small input shape"

def test_create_model_with_large_number_of_labels():
    """Test the function with a large number of labels."""
    input_shape = (40, 100, 1)
    num_labels = 100
    model = create_model(input_shape, num_labels)
    assert model is not None, "Model creation failed with large number of labels"

def test_create_model_returns_different_model_instance_for_same_params():
    """Test that the function returns a different model instance for the same parameters when called multiple times."""
    input_shape = (40, 100, 1)
    num_labels = 10
    model1 = create_model(input_shape, num_labels)
    model2 = create_model(input_shape, num_labels)
    assert model1 is not model2, "Model instances are the same, should be different"

def test_create_model_output_layer_shape():
    """Check that the output layer of the model has the expected number of nodes based on num_labels."""
    input_shape = (40, 100, 1)
    num_labels = 10
    model = create_model(input_shape, num_labels)

    # Attempt to predict with dummy data to inspect the output shape - Assuming TF backend for now
    try:
        dummy_input = np.random.rand(1, *input_shape)
        output = model.predict(dummy_input)
        assert output.shape[-1] == num_labels, "Output layer shape does not match num_labels"
    except Exception as e:
        pytest.fail(f"Failed to check model output shape: {e}")
