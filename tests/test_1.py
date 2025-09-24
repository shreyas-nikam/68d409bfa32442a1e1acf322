import pytest
import numpy as np
from definition_f10f56790df54f28a228ce1ec8743d46 import preprocess_audio

@pytest.fixture
def mock_audio_data():
    return np.random.rand(1000)

@pytest.mark.parametrize("sample_rate", [22050, 44100])
def test_preprocess_audio_valid_input(mock_audio_data, sample_rate):
    mfccs = preprocess_audio(mock_audio_data, sample_rate)
    assert mfccs is not None
    assert isinstance(mfccs, np.ndarray)
    # Basic check that MFCCs were extracted

def test_preprocess_audio_empty_audio_data():
    with pytest.raises(Exception):
        preprocess_audio(np.array([]), 22050)  # Librosa might throw an exception if audio_data is empty

def test_preprocess_audio_invalid_audio_data_type():
    with pytest.raises(TypeError):
        preprocess_audio("invalid", 22050)

def test_preprocess_audio_zero_sample_rate(mock_audio_data):
     with pytest.raises(Exception): # Librosa will likely error with sr <= 0, use broad exception to cover librosa's specific error.
        preprocess_audio(mock_audio_data, 0)

def test_preprocess_audio_negative_sample_rate(mock_audio_data):
    with pytest.raises(Exception): # Librosa will likely error with sr <= 0, use broad exception to cover librosa's specific error.
        preprocess_audio(mock_audio_data, -1)
