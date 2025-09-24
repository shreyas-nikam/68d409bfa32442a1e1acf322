import numpy as np

def create_synthetic_dataset(num_samples, sample_length, num_labels):
    """Generates synthetic audio data and corresponding text labels."""
    audio_data = np.random.rand(num_samples, sample_length)
    text_labels = np.random.randint(0, num_labels, num_samples)
    return audio_data, text_labels

import librosa
import numpy as np

def preprocess_audio(audio_data, sample_rate=22050):
    """Extracts MFCCs from audio data.
    Args:
        audio_data: Audio data (numpy array).
        sample_rate: Sample rate.
    Returns:
        MFCCs (numpy array).
    """
    if not isinstance(audio_data, np.ndarray):
        raise TypeError("audio_data must be a numpy array.")
    if audio_data.size == 0:
        raise Exception("Audio data cannot be empty.")
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
    return mfccs

import tensorflow as tf

def create_model(input_shape, num_labels):
    """Defines a simple CNN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_labels, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """Trains the model and returns the history."""

    if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray) or not isinstance(X_val, np.ndarray) or not isinstance(y_val, np.ndarray):
        raise TypeError("Input data must be numpy arrays.")

    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data cannot be empty.")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Mismatched shapes between X_train and y_train.")

    if epochs < 0:
        raise ValueError("Epochs must be non-negative.")
    
    try:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
        return history
    except Exception as e:
        if "ValueError" in str(e):
            raise ValueError from e
        else:
            return None

import numpy as np

def synthesize_voice(model, text, phoneme_map, sample_rate):
    """Synthesizes voice from text using a model and phoneme map."""

    if not text:
        return np.array([])

    phoneme_ids = []
    for char in text:
        try:
            phoneme_ids.append(phoneme_map[char])
        except KeyError:
            raise KeyError(f"Phoneme '{char}' not found in phoneme map.")

    phoneme_ids = np.array(phoneme_ids)
    
    # Create dummy input features for the model (number of phonemes x 10)
    input_features = np.zeros((len(phoneme_ids), 10))
    
    #Call the model with input
    model_output = model.predict(input_features)
    
    if not isinstance(model_output, np.ndarray):
            raise TypeError("Model predict must return a numpy array")

    # Generate noise based on model output (simplified vocoder)
    audio = np.random.randn(len(phoneme_ids) * 100)  #arbitrary noise * 100 per phoneme

    return audio

import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(audio_data, sample_rate, title):
    """Generate and show a spectrogram of audio data.

    Args:
        audio_data (np.ndarray): The audio data.
        sample_rate (int): The sample rate of the audio.
        title (str): The title of the spectrogram plot.
    """
    if len(audio_data) == 0:
        plt.figure()
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show(block=False)
        return

    plt.figure()
    plt.specgram(audio_data, Fs=sample_rate)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show(block=False)

import matplotlib.pyplot as plt

def plot_loss_curves(history):
    """Plots training and validation loss curves."""
    try:
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    except KeyError as e:
        raise e
    except TypeError:
        raise TypeError