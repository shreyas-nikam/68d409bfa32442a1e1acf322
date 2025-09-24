import pytest
import numpy as np
from definition_eb230f84b5754e5fb0270b4f398dacad import preprocess_audio

# Pre-calculated values for expected time_frames based on librosa's default parameters
# For sample_length=22050, sr=22050, n_mfcc=40, default hop_length=512, frames are typically 44.
_EXPECTED_TIME_FRAMES_22050_SR = 44
# For sample_length=11025, sr=11025, n_mfcc=40, default hop_length=512, frames are typically 22.
_EXPECTED_TIME_FRAMES_11025_SR = 22
_N_MFCC = 40

@pytest.mark.parametrize(
    "audio_data_input, sample_rate_input, expected_output",
    [
        # Test Case 1: Happy Path - Multiple valid audio samples
        # Expected: A 3D numpy array of shape (num_samples, n_mfcc, time_frames)
        (np.random.rand(5, 22050).astype(np.float32), 22050, (5, _N_MFCC, _EXPECTED_TIME_FRAMES_22050_SR)),

        # Test Case 2: Edge Case - Empty audio_data (0 samples)
        # The function should still return a 3D array, but with 0 samples dimension.
        # The last dimension (time_frames) is based on the sample_length that was implicitly provided.
        (np.empty((0, 22050), dtype=np.float32), 22050, (0, _N_MFCC, _EXPECTED_TIME_FRAMES_22050_SR)),

        # Test Case 3: Edge Case - Single audio sample
        # Expected: A 3D numpy array of shape (1, n_mfcc, time_frames)
        (np.random.rand(1, 11025).astype(np.float32), 11025, (1, _N_MFCC, _EXPECTED_TIME_FRAMES_11025_SR)),

        # Test Case 4: Error Case - Invalid audio_data type (e.g., list of lists instead of numpy array)
        # librosa expects a numpy array for audio processing. Iterating over a list of lists
        # will cause a TypeError when `librosa.feature.mfcc` is called with a list.
        ([[0.1, 0.2], [0.3, 0.4]], 22050, TypeError),

        # Test Case 5: Error Case - Invalid sample_rate (non-positive)
        # librosa's MFCC function requires a positive sampling rate, otherwise it raises a ValueError.
        (np.random.rand(1, 22050).astype(np.float32), 0, ValueError),
    ]
)
def test_preprocess_audio(audio_data_input, sample_rate_input, expected_output):
    if isinstance(expected_output, tuple):  # Expected a successful return with a specific shape
        result = preprocess_audio(audio_data_input, sample_rate_input)
        assert isinstance(result, np.ndarray)
        assert result.shape == expected_output
    else:  # Expected an exception
        with pytest.raises(expected_output):
            preprocess_audio(audio_data_input, sample_rate_input)