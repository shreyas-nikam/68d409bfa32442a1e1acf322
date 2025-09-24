import pytest
import numpy as np
import matplotlib.pyplot as plt
from definition_62106b8331fb4951826726368cecbd5d import plot_spectrogram

def test_plot_spectrogram_valid_input():
    # Test with valid audio data and sample rate
    audio_data = np.random.rand(1000)
    sample_rate = 22050
    title = "Test Spectrogram"
    try:
        plot_spectrogram(audio_data, sample_rate, title)
        plt.close()  # Close the plot to prevent display during testing
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_plot_spectrogram_empty_audio():
    # Test with empty audio data
    audio_data = np.array([])
    sample_rate = 22050
    title = "Empty Audio Spectrogram"
    try:
        plot_spectrogram(audio_data, sample_rate, title)
        plt.close()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_plot_spectrogram_invalid_sample_rate():
    # Test with invalid sample rate (zero)
    audio_data = np.random.rand(1000)
    sample_rate = 0
    title = "Invalid Sample Rate"
    with pytest.raises(Exception):
        plot_spectrogram(audio_data, sample_rate, title)
        plt.close()

def test_plot_spectrogram_non_numeric_audio():
    # Test with non-numeric audio data
    audio_data = ["a", "b", "c"]
    sample_rate = 22050
    title = "Non-Numeric Audio"
    with pytest.raises(TypeError):
        plot_spectrogram(audio_data, sample_rate, title)
        plt.close()

def test_plot_spectrogram_large_sample_rate():
     # Test with large sample rate
    audio_data = np.random.rand(1000)
    sample_rate = 48000
    title = "High sample rate"
    try:
        plot_spectrogram(audio_data, sample_rate, title)
        plt.close()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
