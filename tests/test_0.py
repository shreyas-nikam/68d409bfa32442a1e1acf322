import pytest
import numpy as np
from definition_290f69775e464318869609964e5b43fd import create_synthetic_dataset

@pytest.mark.parametrize(
    "num_samples, sample_length, num_labels, expected_audio_shape, expected_labels_shape, expected_min_label_val, expected_max_label_val, expected_exception",
    [
        # Test Case 1: Standard valid input - multiple samples, length, and labels
        (100, 1000, 10, (100, 1000), (100,), 0, 10, None),
        # Test Case 2: Edge case - zero samples
        (0, 100, 5, (0, 100), (0,), 0, 5, None),
        # Test Case 3: Edge case - zero sample length
        (10, 0, 5, (10, 0), (10,), 0, 5, None),
        # Test Case 4: Error case - negative num_samples (invalid value)
        (-5, 100, 5, None, None, None, None, ValueError),
        # Test Case 5: Error case - invalid type for sample_length (TypeError)
        (10, "invalid", 5, None, None, None, None, TypeError),
    ]
)
def test_create_synthetic_dataset(num_samples, sample_length, num_labels, 
                                  expected_audio_shape, expected_labels_shape, 
                                  expected_min_label_val, expected_max_label_val, 
                                  expected_exception):
    if expected_exception:
        # If an exception is expected, assert that the correct exception is raised
        with pytest.raises(expected_exception):
            create_synthetic_dataset(num_samples, sample_length, num_labels)
    else:
        # If no exception is expected, test the function's output
        audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)

        # Assert output types are numpy arrays
        assert isinstance(audio_data, np.ndarray)
        assert isinstance(text_labels, np.ndarray)

        # Assert output shapes
        assert audio_data.shape == expected_audio_shape
        assert text_labels.shape == expected_labels_shape

        # Only check content if there are samples generated
        if num_samples > 0:
            # Assert audio data values are within the expected range [0, 1]
            assert np.all(audio_data >= 0)
            assert np.all(audio_data <= 1)
            assert np.issubdtype(audio_data.dtype, np.floating)

            # Assert text labels are integers and within the expected range [0, num_labels]
            assert np.issubdtype(text_labels.dtype, np.integer)
            assert np.all(text_labels >= expected_min_label_val)
            assert np.all(text_labels <= expected_max_label_val)