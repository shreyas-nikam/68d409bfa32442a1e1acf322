import pytest
import numpy as np
from unittest.mock import MagicMock
from definition_614083da89624905b9f28125521af5d5 import synthesize_voice

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([[0.1, 0.2, 0.7]]) 
    return model

@pytest.mark.parametrize("text, expected_output_shape", [
    ("hello", (30,)),
    ("", (0,)),
    ("a", (10,)),
])
def test_synthesize_voice_basic(mock_model, text, expected_output_shape):
    phoneme_map = {"h": 0, "e": 1, "l": 2, "o": 3, "a": 4}
    sample_rate = 22050
    mock_model.predict.return_value = np.random.rand(len(text), 3)
    audio = synthesize_voice(mock_model, text, phoneme_map, sample_rate)
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (len(text) * 10,)

def test_synthesize_voice_unknown_phoneme(mock_model):
    text = "xyz"
    phoneme_map = {"x": 0, "y": 1}
    sample_rate = 22050
    with pytest.raises(KeyError):
        synthesize_voice(mock_model, text, phoneme_map, sample_rate)

def test_synthesize_voice_empty_phoneme_map(mock_model):
    text = "hello"
    phoneme_map = {}
    sample_rate = 22050
    with pytest.raises(KeyError):
        synthesize_voice(mock_model, text, phoneme_map, sample_rate)

def test_synthesize_voice_different_output_len(mock_model):
    text = "hello"
    phoneme_map = {"h": 0, "e": 1, "l": 2, "o": 3}
    sample_rate = 22050
    mock_model.predict.return_value = np.random.rand(len(text), 5)
    audio = synthesize_voice(mock_model, text, phoneme_map, sample_rate)
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (len(text) * 10,)
