import pytest
import numpy as np
from definition_c2e51e19fb27428885cc5027e1265a2d import create_synthetic_dataset

def test_create_synthetic_dataset_positive_values():
    num_samples = 10
    sample_length = 5
    num_labels = 3
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert audio_data.shape == (num_samples, sample_length)
    assert text_labels.shape == (num_samples,)
    assert np.all(audio_data >= 0)
    assert np.all(audio_data <= 1)
    assert np.all(text_labels >= 0)
    assert np.all(text_labels < num_labels)

def test_create_synthetic_dataset_zero_samples():
    num_samples = 0
    sample_length = 5
    num_labels = 3
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert audio_data.shape == (num_samples, sample_length)
    assert text_labels.shape == (num_samples,)

def test_create_synthetic_dataset_zero_length():
    num_samples = 10
    sample_length = 0
    num_labels = 3
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert audio_data.shape == (num_samples, sample_length)
    assert text_labels.shape == (num_samples,)

def test_create_synthetic_dataset_zero_labels():
    num_samples = 10
    sample_length = 5
    num_labels = 0
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert audio_data.shape == (num_samples, sample_length)
    assert text_labels.shape == (num_samples,)
    assert np.all(text_labels >= 0)
    assert np.all(text_labels < num_labels)

def test_create_synthetic_dataset_large_values():
    num_samples = 100
    sample_length = 1000
    num_labels = 50
    audio_data, text_labels = create_synthetic_dataset(num_samples, sample_length, num_labels)
    assert audio_data.shape == (num_samples, sample_length)
    assert text_labels.shape == (num_samples,)
    assert np.all(audio_data >= 0)
    assert np.all(audio_data <= 1)
    assert np.all(text_labels >= 0)
    assert np.all(text_labels < num_labels)
