import pytest
import numpy as np
from definition_42b556685bfd46d49ee71f931910087d import create_synthetic_dataset

@pytest.mark.parametrize("num_samples, sample_length, num_labels, expected_audio_shape, expected_labels_shape", [
    (10, 100, 5, (10, 100), (10,)),
    (5, 200, 10, (5, 200), (5,)),
    (1, 50, 2, (1, 50), (1,)),
])
def test_create_synthetic_dataset_shapes(num_samples, sample_length, num_labels, expected_audio_shape, expected_labels_shape):
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert audio_data.shape == expected_audio_shape
    assert text_labels.shape == expected_labels_shape

@pytest.mark.parametrize("num_samples, sample_length, num_labels", [
    (10, 100, 5),
    (5, 200, 10),
    (1, 50, 2),
])
def test_create_synthetic_dataset_data_types(num_samples, sample_length, num_labels):
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert isinstance(audio_data, np.ndarray)
    assert isinstance(text_labels, np.ndarray)
    assert audio_data.dtype == np.float64  # Assuming default numpy float type
    assert text_labels.dtype == np.int64

def test_create_synthetic_dataset_num_labels_range():
    num_samples = 10
    sample_length = 100
    num_labels = 5
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert np.all(text_labels >= 0)
    assert np.all(text_labels < num_labels)

def test_create_synthetic_dataset_audio_data_range():
    num_samples = 10
    sample_length = 100
    num_labels = 5
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert np.all(audio_data >= 0)
    assert np.all(audio_data <= 1)

def test_create_synthetic_dataset_zero_samples():
    num_samples = 0
    sample_length = 100
    num_labels = 5
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert audio_data.shape == (0, sample_length)
    assert text_labels.shape == (0,)
