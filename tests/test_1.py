import pytest
import numpy as np
import librosa
from definition_dde44d480258423fb0d96393db80011b import preprocess_audio

@pytest.fixture
def mock_audio_data():
    return np.random.rand(1000)

def test_preprocess_audio_valid_input(mock_audio_data):
    mfccs = preprocess_audio(mock_audio_data, sample_rate=22050)
    assert isinstance(mfccs, np.ndarray)
    assert mfccs.shape[0] == 20  # Default n_mfcc = 20 (from librosa default)
    assert mfccs.shape[1] > 0

def test_preprocess_audio_default_sample_rate(mock_audio_data):
    mfccs = preprocess_audio(mock_audio_data)
    assert isinstance(mfccs, np.ndarray)

def test_preprocess_audio_empty_audio_data():
    with pytest.raises(Exception):  # Expecting librosa to raise an exception with empty input
        preprocess_audio(np.array([]), sample_rate=22050)

def test_preprocess_audio_invalid_audio_data_type():
    with pytest.raises(TypeError):
        preprocess_audio("invalid_audio_data", sample_rate=22050)

def test_preprocess_audio_low_sample_rate(mock_audio_data):
    mfccs = preprocess_audio(mock_audio_data, sample_rate=1000)
    assert isinstance(mfccs, np.ndarray)

