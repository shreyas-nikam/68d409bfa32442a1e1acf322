import pytest
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# definition_fae169bee29349bf8495dcd290534b62
from definition_fae169bee29349bf8495dcd290534b62 import plot_spectrogram
# </your_module>

@pytest.mark.parametrize(
    "audio_data_input, sample_rate_input, title_input, expected_outcome",
    [
        # Test Case 1: Valid input. Expected outcome is None (successful execution).
        (np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 22050, "Valid Spectrogram", None),
        # Test Case 2: Empty audio data. Expected outcome is ValueError from librosa.
        (np.array([]), 22050, "Empty Audio", ValueError),
        # Test Case 3: Invalid audio_data type (e.g., string). Expected outcome is TypeError.
        ("not an array", 22050, "Invalid Data Type", TypeError),
        # Test Case 4: Invalid sample_rate type (e.g., string). Expected outcome is TypeError.
        (np.array([0.1, 0.2]), "invalid_sr", "Invalid SR Type", TypeError),
        # Test Case 5: Zero sample_rate. Expected outcome is librosa.util.exceptions.ParameterError.
        (np.array([0.1, 0.2]), 0, "Zero SR", librosa.util.exceptions.ParameterError),
    ],
)
def test_plot_spectrogram(
    mocker, audio_data_input, sample_rate_input, title_input, expected_outcome
):
    # Mock display and plotting functions that plot_spectrogram should call
    mock_specshow = mocker.patch("librosa.display.specshow")
    mock_title = mocker.patch("matplotlib.pyplot.title")
    mock_colorbar = mocker.patch("matplotlib.pyplot.colorbar")
    mock_tight_layout = mocker.patch("matplotlib.pyplot.tight_layout")
    mock_show = mocker.patch("matplotlib.pyplot.show")

    if expected_outcome is None:
        # This branch covers successful execution.
        # Mock librosa's internal feature extraction to avoid actual heavy computation
        # and ensure these functions are called with correct arguments.
        mock_melspectrogram = mocker.patch("librosa.feature.melspectrogram", return_value=np.ones((10, 10)))
        mock_power_to_db = mocker.patch("librosa.power_to_db", return_value=np.ones((10, 10)))

        # Call the function
        result = plot_spectrogram(audio_data_input, sample_rate_input, title_input)

        # Assert that the function returns None as it's a display function
        assert result is None

        # Verify that librosa processing and display functions were called
        mock_melspectrogram.assert_called_once_with(y=audio_data_input, sr=sample_rate_input)
        mock_power_to_db.assert_called_once()
        mock_specshow.assert_called_once()

        # Check specific keyword arguments passed to specshow (e.g., sample_rate, axis labels)
        call_kwargs = mock_specshow.call_args[1]
        assert call_kwargs.get('sr') == sample_rate_input
        # Based on notebook spec, assuming typical x_axis='time' and y_axis='mel' for spectrograms
        assert call_kwargs.get('x_axis') == 'time'
        assert call_kwargs.get('y_axis') == 'mel'

        # Verify matplotlib plotting functions were called with correct arguments
        mock_title.assert_called_once_with(title_input)
        mock_colorbar.assert_called_once_with(format="%+2.0f dB") # Standard format from notebook spec
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()

    else:
        # This branch covers expected exception cases.
        # For these cases, we do NOT mock librosa.feature.melspectrogram or librosa.power_to_db,
        # allowing the actual librosa functions to be called and raise their respective exceptions.
        # The mocks for display/plotting functions ensure they are NOT called if an error occurs earlier.

        with pytest.raises(expected_outcome):
            plot_spectrogram(audio_data_input, sample_rate_input, title_input)

        # Assert that none of the display/plotting functions were called if an exception occurred
        mock_specshow.assert_not_called()
        mock_title.assert_not_called()
        mock_colorbar.assert_not_called()
        mock_tight_layout.assert_not_called()
        mock_show.assert_not_called()