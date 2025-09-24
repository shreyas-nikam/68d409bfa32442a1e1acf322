import numpy as np
import librosa
import librosa.display
import plotly.graph_objects as go
import streamlit as st
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def create_synthetic_dataset(num_samples, sample_length, num_labels):
    """
    Generates synthetic audio data and corresponding numerical text labels.
    Audio data is simple sine waves with varying frequencies.
    """
    audio_data = np.zeros((num_samples, sample_length), dtype=np.float32)
    text_labels = np.random.randint(0, num_labels, num_samples)

    for i in range(num_samples):
        frequency = 100 + text_labels[i] * 50  # Vary frequency based on label
        t = np.linspace(0, 1, sample_length, endpoint=False)
        audio_data[i, :] = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio_data, text_labels

def preprocess_audio(audio_data, sample_rate, n_mfcc=40, sample_length=None):
    """
    Extracts MFCCs from a batch of audio data.
    Assumes audio_data is a 2D numpy array (num_samples, sample_length).
    Dynamically calculates n_fft and hop_length based on sample_length.
    Ensures all MFCC arrays have a consistent number of frames.
    """
    if sample_length is None:
        sample_length = audio_data.shape[1] if audio_data.shape[1] > 0 else 1000

    n_fft = max(1, int(2**np.floor(np.log2(sample_length))))
    hop_length = n_fft // 4
    if hop_length == 0 and n_fft > 0: # Ensure hop_length is at least 1 if n_fft is positive
        hop_length = 1
    elif n_fft == 0: # If n_fft is 0, hop_length must also be 0
        hop_length = 0

    # Calculate the target number of frames for consistency
    target_frames = 0
    if sample_length >= n_fft:
        target_frames = 1 + (sample_length - n_fft) // hop_length
    # Ensure target_frames is at least 1 if there's audio data to process
    if target_frames == 0 and sample_length > 0: # If sample_length is too small for n_fft/hop_length to yield frames, force at least 1
        target_frames = 1

    target_frames = max(target_frames, 4)
    
    mfccs_list = []
    for i in range(audio_data.shape[0]):
        if np.sum(np.abs(audio_data[i, :])) > 0:
            mfccs = librosa.feature.mfcc(y=audio_data[i, :], sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            
            # Enforce consistent number of frames
            if mfccs.shape[1] < target_frames:
                # Pad if shorter
                pad_width = target_frames - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            elif mfccs.shape[1] > target_frames:
                # Truncate if longer
                mfccs = mfccs[:, :target_frames]
            
            mfccs_list.append(mfccs)
        else:
            # For silent audio, append an MFCC array with target_frames
            mfccs_list.append(np.zeros((n_mfcc, target_frames), dtype=np.float32))

    if mfccs_list:
        return np.array(mfccs_list)
    else:
        # Return empty array with correct dimensions if no audio data was processed
        return np.empty((0, n_mfcc, target_frames), dtype=np.float32)

def plot_spectrogram_plotly(audio, sample_rate, title):
    """
    Generates a Plotly spectrogram figure for a single audio sample.
    """
    if np.sum(np.abs(audio)) == 0:
        st.warning("Cannot plot spectrogram for silent audio.")
        return go.Figure()

    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    fig = go.Figure(data=go.Heatmap(
                   z=D,
                   y=librosa.fft_frequencies(sr=sample_rate),
                   x=librosa.frames_to_time(np.arange(D.shape[1]), sr=sample_rate),
                   colorscale='Viridis'))
    fig.update_layout(title=title,
                      yaxis_title='Frequency (Hz)',
                      xaxis_title='Time (s)')
    return fig

def plot_mfccs_plotly(mfccs, sample_rate, title):
    """
    Generates a Plotly MFCCs figure for a single MFCC array.
    """
    if mfccs.shape[1] == 0:
        st.warning("Cannot plot MFCCs for empty data.")
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
                   z=mfccs,
                   colorscale='Viridis'))
    fig.update_layout(title=title,
                      yaxis_title='MFCC Coefficient',
                      xaxis_title='Frame')
    return fig

def create_model(input_shape, num_labels):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """
    Trains the given Keras model.
    """
    # Ensure y_train and y_val are one-hot encoded
    y_train_encoded = to_categorical(y_train, num_classes=model.output_shape[-1])
    y_val_encoded = to_categorical(y_val, num_classes=model.output_shape[-1])

    # Reshape X_train and X_val to include a channel dimension
    # Assuming MFCCs are (num_samples, n_mfcc, num_frames)
    # Keras Conv2D expects (batch, height, width, channels)
    X_train_reshaped = np.expand_dims(X_train, -1)
    X_val_reshaped = np.expand_dims(X_val, -1)

    history = model.fit(X_train_reshaped, y_train_encoded,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val_reshaped, y_val_encoded),
                        verbose=0)
    return history

def plot_loss_curves_plotly(history):
    """
    Generates a Plotly figure for training and validation loss/accuracy.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'],
                             mode='lines',
                             name='Training Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'],
                             mode='lines',
                             name='Validation Loss'))
    fig.add_trace(go.Scatter(y=history.history['accuracy'],
                             mode='lines',
                             name='Training Accuracy'))
    fig.add_trace(go.Scatter(y=history.history['val_accuracy'],
                             mode='lines',
                             name='Validation Accuracy'))
    fig.update_layout(title='Model Loss and Accuracy',
                      xaxis_title='Epoch',
                      yaxis_title='Value')
    return fig

def print_model_summary(model):
    """
    Returns a string summary of the Keras model.
    """
    with io.StringIO() as stream:
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        output = stream.getvalue()
    return output
