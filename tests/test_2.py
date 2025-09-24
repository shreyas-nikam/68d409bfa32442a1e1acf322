import pytest
import tensorflow as tf
from definition_ff2fca5271f64a268e014b1499a688a2 import create_model

@pytest.mark.parametrize(
    "input_shape, num_labels, expected",
    [
        # Test case 1: Valid input for a standard model (expected functionality)
        ((40, 100, 1), 10, (None, 10)),
        # Test case 2: Valid input with a single output label (edge case for num_labels)
        ((40, 100, 1), 1, (None, 1)),
        # Test case 3: Invalid input_shape type (list instead of tuple)
        ([40, 100, 1], 10, TypeError),
        # Test case 4: Invalid num_labels type (string instead of int)
        ((40, 100, 1), "ten", TypeError),
        # Test case 5: num_labels is zero (invalid value, as a classification model needs positive classes)
        ((40, 100, 1), 0, ValueError),
    ]
)
def test_create_model(input_shape, num_labels, expected):
    try:
        model = create_model(input_shape, num_labels)
        # If no exception, assert it's a Keras Model and check its output shape
        assert isinstance(model, tf.keras.Model)
        assert model.output_shape == expected
        # Optionally, check the activation of the last layer
        assert model.layers[-1].activation.__name__ == 'softmax'
    except Exception as e:
        # If an exception occurred, assert its type matches the expected exception
        assert isinstance(e, expected)