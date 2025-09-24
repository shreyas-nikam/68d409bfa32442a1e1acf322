import pytest
import numpy as np
from definition_9008e8e4093b45e69a2a890f5fc2ac60 import synthesize_voice

# Mock model class for testing
class MockModel:
    def predict(self, text_input):
        # A very simplified mock: model 'predicts' the same labels it received.
        # This simulates the model confirming or processing the input labels
        # before they are used with the phoneme_map for synthesis.
        if isinstance(text_input, int):
            return np.array([text_input])
        return np.array(text_input)

# Define mock audio snippets
_audio_snippet_a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
_audio_snippet_b = np.array([0.4, 0.5, 0.6], dtype=np.float32)
_audio_snippet_c = np.array([0.7, 0.8, 0.9], dtype=np.float32)

@pytest.fixture
def mock_model_instance():
    """Provides a mock neural network model."""
    return MockModel()

@pytest.fixture
def mock_phoneme_map_data():
    """Provides a mock phoneme map."""
    return {
        0: _audio_snippet_a,
        1: _audio_snippet_b,
        2: _audio_snippet_c,
    }

@pytest.fixture
def default_sample_rate():
    """Provides a default sample rate."""
    return 16000

# Test 1: Basic functionality with a list of numerical labels
def test_synthesize_voice_list_labels(mock_model_instance, mock_phoneme_map_data, default_sample_rate):
    """
    Tests synthesis of voice from a list of numerical text labels.
    Expects concatenation of corresponding audio snippets.
    """
    text_input = [0, 1]
    expected_audio = np.concatenate([mock_phoneme_map_data[0], mock_phoneme_map_data[1]])
    
    synthesized_audio = synthesize_voice(mock_model_instance, text_input, mock_phoneme_map_data, default_sample_rate)
    np.testing.assert_array_almost_equal(synthesized_audio, expected_audio)

# Test 2: Basic functionality with a single numerical label
def test_synthesize_voice_single_label(mock_model_instance, mock_phoneme_map_data, default_sample_rate):
    """
    Tests synthesis of voice from a single numerical text label.
    Expects the corresponding single audio snippet.
    """
    text_input = 0
    expected_audio = mock_phoneme_map_data[0]
    
    synthesized_audio = synthesize_voice(mock_model_instance, text_input, mock_phoneme_map_data, default_sample_rate)
    np.testing.assert_array_almost_equal(synthesized_audio, expected_audio)

# Test 3: Edge case: Empty list for text input
def test_synthesize_voice_empty_text(mock_model_instance, mock_phoneme_map_data, default_sample_rate):
    """
    Tests synthesis with an empty list of text labels.
    Expects an empty NumPy array as output.
    """
    text_input = []
    expected_audio = np.array([], dtype=np.float32) # Ensure dtype matches expected audio snippets
    
    synthesized_audio = synthesize_voice(mock_model_instance, text_input, mock_phoneme_map_data, default_sample_rate)
    np.testing.assert_array_almost_equal(synthesized_audio, expected_audio)

# Test 4: Edge case: Text contains labels not present in phoneme_map
def test_synthesize_voice_missing_phoneme_map_entry(mock_model_instance, mock_phoneme_map_data, default_sample_rate):
    """
    Tests behavior when a text label is not found in the phoneme_map.
    Expects a KeyError to be raised.
    """
    text_input = [0, 99, 1] # 99 is not in mock_phoneme_map_data
    
    with pytest.raises(KeyError):
        synthesize_voice(mock_model_instance, text_input, mock_phoneme_map_data, default_sample_rate)

# Test 5: Invalid input types for text, phoneme_map, or sample_rate
@pytest.mark.parametrize("text_input, phoneme_map_input, sample_rate_input, expected_exception", [
    # Invalid text type: string instead of int or list[int]
    ("hello", {0: _audio_snippet_a}, 16000, TypeError),
    # Invalid phoneme_map type: string instead of dict
    ([0, 1], "not_a_dict", 16000, TypeError),
    # Invalid sample_rate type: string instead of int
    ([0, 1], {0: _audio_snippet_a}, "wrong_rate", TypeError),
    # Invalid text type: None instead of int or list[int]
    (None, {0: _audio_snippet_a}, 16000, TypeError),
])
def test_synthesize_voice_invalid_input_types(mock_model_instance, text_input, phoneme_map_input, sample_rate_input, expected_exception):
    """
    Tests that the function raises TypeError for various invalid input types.
    """
    with pytest.raises(expected_exception):
        synthesize_voice(mock_model_instance, text_input, phoneme_map_input, sample_rate_input)