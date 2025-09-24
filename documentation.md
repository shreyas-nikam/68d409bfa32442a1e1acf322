id: 68d409bfa32442a1e1acf322_documentation
summary: study on different Text-to-Speech technologies Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Building a Custom Voice: A Text-to-Speech Codelab

This codelab guides you through the process of building a simplified Text-to-Speech (TTS) system. You'll learn about data generation, audio preprocessing, model training, and voice synthesis. The application uses synthetic data to simplify the process and focus on core concepts. By the end of this codelab, you'll have a hands-on understanding of how a TTS system works.

## Introduction
Duration: 00:05

This application demonstrates the core principles behind custom voice creation, even without the complexities of real-world audio. We will build a simplified Text-to-Speech (TTS) system.

The primary goals of this lab are:

*   **Data Generation**: Creating synthetic audio data and corresponding text labels.
*   **Preprocessing**: Transforming raw audio into meaningful features using Mel-Frequency Cepstral Coefficients (MFCCs).
*   **Model Training**: Training a basic neural network model to map audio features to text labels.
*   **Voice Synthesis**: Synthesizing voice from new text inputs using the trained model.

You will have the opportunity to interact with various parameters and observe their impact on the data, model, and synthesized output.

## Setting up the Environment and Understanding the Application Structure
Duration: 00:10

Before diving into the code, it's essential to understand the structure of the Streamlit application. Here's a breakdown:

*   **`app.py`**: This is the main entry point of the Streamlit application. It handles the overall structure, navigation, and calls the functions from the other pages. It uses `st.sidebar` to provide navigation between different sections of the codelab.
*   **`application_pages/page1.py`**: This script contains the code for data generation and preprocessing.  It allows you to generate synthetic audio data and extract MFCC features, visualizing the data and MFCCs.
*   **`application_pages/page2.py`**: This script focuses on model training. You'll split the data into training and validation sets, define and train a neural network model, and visualize the training progress using loss curves.
*   **`application_pages/page3.py`**: This script implements voice synthesis and provides an interactive environment to experiment with different training parameters.  You'll synthesize audio from input labels, compare original and synthesized audio, and adjust parameters to observe their impact on the synthesized voice.
*   **`application_pages/utils.py`**: This script contains utility functions used throughout the application, such as creating synthetic datasets, preprocessing audio, plotting spectrograms and MFCCs, defining and training the model, and synthesizing voice.

<aside class="positive">
The modular structure of the application makes it easier to understand and modify each component independently.
</aside>

## Data Generation & Preprocessing
Duration: 00:20

This section focuses on creating and preparing the data for our TTS model.  We'll use the functions defined in `application_pages/utils.py` to generate synthetic audio and extract MFCC features.

1.  **Navigate to the "Data Generation & Preprocessing" page:** Use the sidebar in the Streamlit application to navigate to this page.

2.  **Synthetic Data Generation:**

    *   The page presents sliders to control the characteristics of the synthetic dataset:
        *   **Number of Samples**:  The total number of synthetic audio files that will be generated.
        *   **Sample Length**: The length (in data points) of each individual audio sample.
        *   **Number of Labels**:  The number of unique text labels that will be associated with the audio samples.

    *   Modify these sliders and observe the changes in the output:
        *   `audio_data.shape` reflects the dimensions (number of samples, length of each sample) of the generated audio dataset.
        *   `text_labels.shape` reflects the number of labels generated.
        *   The audio player shows an example of what the generated synthetic audio sounds like.

    The `create_synthetic_dataset` function in `application_pages/utils.py` handles this data generation:

    ```python
    @st.cache_resource
    def create_synthetic_dataset(num_samples, sample_length, num_labels):
        """
        Generates synthetic audio data and corresponding text labels.
        """
        audio_data = np.random.rand(num_samples, sample_length).astype(np.float32)
        text_labels = np.random.randint(low=0, high=num_labels, size=num_samples)
        return audio_data, text_labels
    ```

    This function generates random audio data and assigns random integer labels to it.

3.  **Preprocessing: MFCC Extraction:**

    *   This section uses a fixed sample rate of 22050 Hz.

    *   The `preprocess_audio` function from `application_pages/utils.py` extracts MFCCs:

    ```python
    @st.cache_resource
    def preprocess_audio(audio_data, sample_rate):
        """
        Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the provided audio data using the librosa library.
        """
        n_mfcc = 40
        num_samples = audio_data.shape[0]
        all_mfccs = []
        for i in range(num_samples):
            current_audio_sample = audio_data[i, :]
            mfccs = librosa.feature.mfcc(y=current_audio_sample, sr=sample_rate, n_mfcc=n_mfcc)
            all_mfccs.append(mfccs)
        return np.stack(all_mfccs)
    ```

    This function iterates through the audio samples, extracts MFCCs using `librosa.feature.mfcc`, and stacks them into a 3D array. `n_mfcc = 40` sets the number of MFCC coefficients to be extracted.

    *   `mfccs_data.shape` displays the shape of the resulting MFCC data. It will be (number of samples, n_mfcc, time frames).

4.  **Visualizing Audio Data and MFCCs:**

    *   This section visualizes the generated audio data and the extracted MFCCs.

    *   Use the "Select Sample Index for Visualization" slider to choose a sample.

    *   The `plot_spectrogram_plotly` and `plot_mfccs_plotly` functions in `application_pages/utils.py` are used for visualization:

    ```python
    def plot_spectrogram_plotly(audio_data, sample_rate, title):
        """
        Generates and displays a spectrogram of the provided audio data using Plotly.
        """
        S = librosa.feature.melspectrogram(y=audio_data.astype(np.float32), sr=sample_rate)
        S_dB = librosa.power_to_db(S, ref=np.max)
        times = librosa.times_like(S_dB, sr=sample_rate)
        mel_freqs = librosa.mel_frequencies(n_mels=S_dB.shape[0], fmax=sample_rate/2)

        fig = go.Figure(data=go.Heatmap(
            z=S_dB,
            x=times,
            y=mel_freqs,
            colorscale='Viridis',
            colorbar=dict(title='dB', titleside='right')
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Mel Frequency (Hz)",
            height=400,
            width=700
        )
        return fig
    ```

    This function calculates the Mel spectrogram, converts it to decibels, and displays it as a heatmap using Plotly.

    ```python
    def plot_mfccs_plotly(mfccs_data, sample_rate, title):
        """
        Generates and displays MFCCs of the provided audio data using Plotly.
        """
        times = librosa.times_like(mfccs_data, sr=sample_rate)
        mfcc_coeffs = np.arange(1, mfccs_data.shape[0] + 1)

        fig = go.Figure(data=go.Heatmap(
            z=mfccs_data,
            x=times,
            y=mfcc_coeffs,
            colorscale='Viridis',
            colorbar=dict(title='Coefficient Value', titleside='right')
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="MFCC Coefficient",
            height=400,
            width=700
        )
        return fig
    ```

    This function displays the MFCC data as a heatmap using Plotly.

<aside class="positive">
Understanding the shape of your data and visualizing it are crucial steps in any machine learning project. Spectrograms and MFCCs provide valuable insights into the characteristics of audio data.
</aside>

## Model Training
Duration: 00:25

In this section, we will define, train, and evaluate a neural network model to map MFCC features to text labels.

1.  **Navigate to the "Model Training" page:** Use the sidebar to navigate to this page. Make sure you have already generated data on the previous page, or this page will not work.

2.  **Data Splitting:**

    *   The MFCC data (`X`) and text labels (`y`) are split into training and validation sets using `train_test_split` from `sklearn.model_selection`.

    *   The split is 80% training and 20% validation, with `random_state=42` for reproducibility.  Stratification is used to maintain label proportions in the training and validation sets, if the number of unique labels is greater than 1.

    *   The shapes of the resulting `X_train`, `X_val`, `y_train`, and `y_val` are displayed.

3.  **Neural Network Model Definition and Training:**

    *   The `create_model` function in `application_pages/utils.py` defines a sequential neural network model:

    ```python
    @st.cache_resource
    def create_model(input_shape, num_labels):
        """
        Defines a simplified neural network model for voice synthesis.
        """
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_labels, activation='softmax')
        ])
        return model
    ```

    This model consists of convolutional layers, max pooling layers, a flatten layer, and dense layers.  The final layer uses a `softmax` activation for multi-class classification. The input shape is determined by the shape of the MFCC data and number of labels by the number of unique labels in the dataset.

    *   The model summary is printed to the Streamlit interface using `st.code`.

    *   The "Number of Epochs" slider allows you to control the number of training epochs.

    *   The `train_model` function in `application_pages/utils.py` trains the model:

    ```python
    @st.cache_resource
    def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
        """
        Trains the defined neural network model on the provided training data and evaluates its performance on the validation data.
        """
        if hasattr(model, 'fit') and callable(model.fit):
            X_train_reshaped = np.expand_dims(X_train, -1)
            X_val_reshaped = np.expand_dims(X_val, -1)

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            history = model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val),
                                epochs=epochs, batch_size=batch_size, verbose=0)
            return history
    ```

    This function compiles the model with the Adam optimizer and sparse categorical crossentropy loss, then trains it using the provided training data and evaluates it on the validation data.  The training history is returned.

4.  **Visualizing Training Progress:**

    *   The `plot_loss_curves_plotly` function in `application_pages/utils.py` plots the training and validation loss curves:

    ```python
    def plot_loss_curves_plotly(history):
        """
        Generates and displays a plot of the training and validation loss curves over epochs using Plotly.
        """
        history_dict = None
        if isinstance(history, dict):
            history_dict = history
        elif hasattr(history, 'history') and isinstance(history.history, dict):
            history_dict = history.history
        else:
            raise AttributeError("Input 'history' must be a dictionary or an object with a 'history' attribute of type dict.")

        train_loss = history_dict.get('loss')
        val_loss = history_dict.get('val_loss')

        fig = go.Figure()

        if train_loss is not None and len(train_loss) > 0:
            epochs = list(range(1, len(train_loss) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Training Loss'))

        if val_loss is not None and len(val_loss) > 0:
            epochs = list(range(1, len(val_loss) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss'))

        fig.update_layout(
            title='Training and Validation Loss',
            xaxis_title='Epochs',
            yaxis_title='Loss',
            legend_title="Metric",
            hovermode="x unified",
            height=500,
            width=700
        )
        return fig
    ```

    This function retrieves the training and validation loss from the training history and displays them as line plots using Plotly.

<aside class="negative">
Pay close attention to the loss curves.  A large gap between the training and validation loss may indicate overfitting.
</aside>

## Voice Synthesis & Interaction
Duration: 00:20

This section focuses on using the trained model to synthesize audio from input labels. It also includes an interactive section to experiment with different training parameters and observe their impact on the synthesized voice.

1.  **Navigate to the "Voice Synthesis & Interaction" page:** Use the sidebar to navigate to this page. Ensure that you have completed the previous sections, or this page will generate an error.

2.  **Synthesize Voice from Input Label:**

    *   The `synthesize_voice` function in `application_pages/utils.py` synthesizes audio from an input label:

    ```python
    @st.cache_resource
    def synthesize_voice(model, text, phoneme_map, sample_rate):
        """
        Converts input text to a sequence of numerical labels, uses the trained model to generate
        corresponding MFCCs, and then reconstructs a synthetic audio waveform.
        """
        if isinstance(text, int):
            predicted_labels = np.array([text])
        elif isinstance(text, list):
            predicted_labels = np.array(text)
        else:
            predicted_labels = text

        audio_snippets = []
        for label in predicted_labels:
            if label not in phoneme_map:
                raise KeyError(f"Phoneme for label {label} not found in map.")
            snippet = phoneme_map[label]
            audio_snippets.append(snippet)

        if not audio_snippets:
            return np.array([], dtype=np.float32)

        synthesized_audio = np.concatenate(audio_snippets)

        return synthesized_audio
    ```

    This function maps the input label to a pre-defined synthetic audio snippet from `phoneme_map` and concatenates the snippets to create the synthesized audio. The phoneme map is defined at the beginning of the page.

    *   The "Enter Input Label (0-9)" number input allows you to specify the label to synthesize.

    *   The synthesized audio is played using `st.audio`, and its duration is displayed.

    *   A spectrogram of the synthesized audio is displayed using `plot_spectrogram_plotly`.

3.  **Comparing Original and Synthesized Audio:**

    *   This section allows you to compare the original synthetic audio snippet for a given label with the audio synthesized by the model for the same label.

    *   The "Select Label for Comparison" slider allows you to choose a label.

    *   The original and synthesized audio are played, and their spectrograms are displayed.

4.  **Interactive Parameter Adjustment:**

    *   This section allows you to experiment with different training parameters and observe their impact on the synthesized voice.

    *   The "Epochs for Interactive Training" slider allows you to adjust the number of training epochs.

    *   The "Input Label(s) for Synthesis (comma-separated 0-9)" text input allows you to specify one or more labels to synthesize, separated by commas.

    *   Clicking the "Run Interactive Synthesis & Retrain" button will retrain the model with the specified number of epochs and synthesize audio for the specified labels.

    *   The training loss curve for the interactive training session is displayed.

    *   The synthesized audio for the specified labels is played, and its spectrogram is displayed.

<aside class="positive">
Experimenting with different training parameters can provide valuable insights into the behavior of the model and the factors that influence the quality of the synthesized voice.
</aside>

## Conclusion
Duration: 00:05

This codelab provided a hands-on experience in building a simplified Text-to-Speech system. You learned about data generation, audio preprocessing, model training, and voice synthesis. You also had the opportunity to interact with various parameters and observe their impact on the synthesized output. This interactive approach aimed to deepen your understanding of the intricate steps involved in crafting a custom TTS experience. Remember that this is a simplified system, and real-world TTS systems involve more complex models, vocoders, and datasets.
