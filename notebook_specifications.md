
## Jupyter Notebook Specification: Custom Voice Creator

### 1. Notebook Overview

This Jupyter Notebook guides users through the process of creating a custom neural voice using their own audio samples.  The notebook will use a simplified synthetic dataset to demonstrate the core concepts without requiring extensive computational resources or real user data upload.

**Learning Goals:**

*   Understand the basic steps involved in creating a custom Text-to-Speech (TTS) voice.
*   Learn how to preprocess audio data for voice model training.
*   Explore the use of a simplified model for voice synthesis.
*   Understand the components of a basic TTS pipeline (preprocessor, encoder, decoder, vocoder - simplified version for demonstration purposes).

### 2. Code Requirements

**Expected Libraries:**

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `scipy`
*   `librosa` (for audio analysis, simplified usage for demonstration)
*   `sklearn` (for data splitting)
*   `tensorflow` or `pytorch` (simplified model implementation, choose one)
*   `IPython.display` (for audio playback)

**Algorithms/Functions to be Implemented:**

1.  **`create_synthetic_dataset(num_samples, sample_length)`:** Generates synthetic audio data and corresponding text labels. The synthetic audio data will be represented as a numpy array of random numbers and the text label will be random words. The sample length will be the number of data point in a single audio sample.
2.  **`preprocess_audio(audio_data, sample_rate)`:** Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the audio data using `librosa`. `sample_rate` defaults to 22050.
3.  **`create_model(input_shape, num_labels)`:** Defines a simplified neural network model.  This model will take MFCCs as input and output a probability distribution over phonemes (represented as numerical labels). A simple model with few layers will work (e.g. with a convolutional layer, a pooling layer, and a dense layer).
4.  **`train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)`:** Trains the defined model on the provided training data and evaluates it on the validation data.
5.  **`synthesize_voice(model, text, phoneme_map, sample_rate)`:** Converts text to phonemes, generates corresponding MFCCs using the model, and reconstructs a synthetic audio waveform.  This will use a very simplified vocoder or directly generate random noise based on the probability distribution output of the model. The `phoneme_map` is a dictionary that maps each phoneme to a unique numerical ID.
6.   **`plot_spectrogram(audio_data, sample_rate, title)`:** Generate and show a spectrogram of the provided audio data.
7.   **`plot_loss_curves(history)`:** Generate and show a plot of the training and validation loss during model training.

**Visualizations:**

1.  Spectrogram of a synthetic audio sample before preprocessing.
2.  Spectrogram of the same audio sample after MFCC extraction.
3.  Plot of training and validation loss during model training.
4.  Interactive audio playback of the original synthetic audio and the synthesized voice.
5.   Bar chart showing the predicted probabilities of different phonemes for a short section of the input text.

### 3. Notebook Sections (Detailed)

**Section 1: Introduction**

*   **Markdown Cell:**
    *   Title: "Custom Voice Creation with Synthetic Data"
    *   Explanation:  Welcome to this Jupyter Notebook, where we'll explore the basics of custom voice creation using a simplified Text-to-Speech (TTS) system. We will use synthetic data to avoid the complexities of real-world audio and focus on the core concepts. This notebook demonstrates the essential steps, including data generation, preprocessing, model training, and voice synthesis. This lab aims to highlight the critical stages involved in crafting a personalized TTS experience.

**Section 2: Library Imports**

*   **Code Cell:**
    *   Imports: Import necessary libraries (`numpy`, `pandas`, `matplotlib`, `scipy`, `librosa`, `sklearn.model_selection`, `tensorflow` or `pytorch`, `IPython.display`).
*   **Markdown Cell:**
    *   Explanation: This cell imports all the Python libraries required for this notebook. Ensure that you have these libraries installed in your environment. You can install them using `pip install [library_name]`.

**Section 3: Synthetic Dataset Generation**

*   **Markdown Cell:**
    *   Title: "Creating a Synthetic Audio Dataset"
    *   Explanation:  To simplify the process and avoid the need for real audio data, we'll generate a synthetic dataset. The dataset will consist of random audio samples and corresponding text labels.  We'll simulate audio data using random numbers and assign arbitrary text labels.  We will generate `num_samples` of audio of length `sample_length` with `num_labels` possible labels.
*   **Code Cell:**
    *   Function: `create_synthetic_dataset(num_samples=100, sample_length=1000, num_labels=10)`: Implement the function to generate the synthetic data. The synthetic audio data will be a numpy array of random numbers between 0 and 1. The text labels will be integers between 0 and `num_labels` (inclusive). The function should return `audio_data` (numpy array of shape `(num_samples, sample_length)`) and `text_labels` (numpy array of shape `(num_samples,)`).
*   **Code Cell:**
    *   Execution: Call `create_synthetic_dataset()` to generate the dataset.
*   **Markdown Cell:**
    *   Explanation: This cell executes the function to create our synthetic dataset and prints the shapes of the generated data. Shapes of `audio_data` and `text_labels` should match expected values.

**Section 4: Audio Preprocessing (MFCC Extraction)**

*   **Markdown Cell:**
    *   Title: "Preprocessing Audio Data: MFCC Extraction"
    *   Explanation: Raw audio data is not suitable for direct input into a neural network. We need to extract meaningful features. Mel-Frequency Cepstral Coefficients (MFCCs) are a common choice for speech recognition and voice synthesis. We'll use `librosa` to extract MFCCs from our synthetic audio samples. The formula for calculating MFCC's is $$MFCC = DCT(log(m)),$$ where $DCT$ stands for Discrete Cosine Transform and $m$ is the power spectrum of the audio after it passes through Mel filters.
*   **Code Cell:**
    *   Function: `preprocess_audio(audio_data, sample_rate=22050)`: Implement the function to extract MFCCs from each audio sample using `librosa.feature.mfcc`.  Set `n_mfcc=40`. The function should return a numpy array of MFCCs of shape `(num_samples, n_mfcc, time_frames)`.
*   **Code Cell:**
    *   Execution: Call `preprocess_audio()` to extract MFCCs from the synthetic audio data.
*   **Markdown Cell:**
    *   Explanation: This cell preprocesses our synthetic audio by extracting MFCCs. The shape of the output MFCCs will depend on the `sample_length` and `sample_rate`.

**Section 5: Visualizing Audio Data (Spectrogram)**

*   **Markdown Cell:**
    *   Title: "Visualizing Audio Data: Spectrogram"
    *   Explanation: To better understand the audio data, we can visualize it as a spectrogram. A spectrogram shows the frequency content of the audio signal over time.  We'll plot the spectrogram before and after MFCC extraction to see the effect of the preprocessing step. The formula for frequency ($f$) can be found by $$f=\frac{1}{T}$$, where $T$ stands for the period.
*   **Code Cell:**
    *   Function: `plot_spectrogram(audio_data, sample_rate, title)`: Implement the function to plot the spectrogram of the audio data using `librosa.display.specshow`.
*   **Code Cell:**
    *   Execution: Select a sample from the `audio_data`, call `plot_spectrogram()` to display its spectrogram before preprocessing, and then call the function again to display the spectrogram of the MFCCs for the same sample after preprocessing.
*   **Markdown Cell:**
    *   Explanation: This section displays the spectrogram of a synthetic audio sample before and after MFCC extraction, allowing us to visualize the effect of the preprocessing.

**Section 6: Splitting Data into Training and Validation Sets**

*   **Markdown Cell:**
    *   Title: "Splitting Data for Training and Validation"
    *   Explanation:  We need to split our dataset into training and validation sets. The training set will be used to train the model, and the validation set will be used to evaluate its performance. A common split is 80% for training and 20% for validation. This helps prevent overfitting and provides a more realistic estimate of the model's performance on unseen data.
*   **Code Cell:**
    *   Import `train_test_split` from `sklearn.model_selection`.
*   **Code Cell:**
    *   Execution: Use `train_test_split` to split the MFCC data (`X`) and text labels (`y`) into training and validation sets (`X_train`, `X_val`, `y_train`, `y_val`).  Set `test_size=0.2` and `random_state=42`.
*   **Markdown Cell:**
    *   Explanation: The data is now split into training and validation sets, ready for model training.

**Section 7: Defining the Model**

*   **Markdown Cell:**
    *   Title: "Defining the Neural Network Model"
    *   Explanation: We'll define a simplified neural network model for voice synthesis. This model will take MFCCs as input and predict phoneme probabilities. The model architecture will consist of a few layers to keep the computational cost low. Select either TensorFlow or PyTorch and stick to it.
*   **Code Cell:**
    *   Function: `create_model(input_shape, num_labels)`: Implement the function to create a simple neural network model. The model will take the shape of the MFCC data (`input_shape`) and the number of unique text labels (`num_labels`) as input parameters.
    *   If using TensorFlow:
        *   Create a `Sequential` model with a few `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
        *   Use `softmax` activation for the final `Dense` layer.
    *   If using PyTorch:
        *   Create a `nn.Module` with similar layers.
        *   Use `nn.LogSoftmax` activation for the final layer.
*   **Code Cell:**
    *   Execution: Call `create_model()` to instantiate the model, passing the appropriate `input_shape` and `num_labels`.
*   **Markdown Cell:**
    *   Explanation:  We have now defined the architecture of our simplified neural network model.

**Section 8: Training the Model**

*   **Markdown Cell:**
    *   Title: "Training the Model"
    *   Explanation: Now we'll train the model using the training data and monitor its performance on the validation data. We'll use an appropriate optimizer (e.g., Adam) and loss function (e.g., categorical cross-entropy).  The goal is to minimize the loss function and improve the model's ability to predict the correct phoneme probabilities for each audio sample.
*   **Code Cell:**
    *   Function: `train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)`: Implement the function to train the model.
        *   If using TensorFlow:
            *   Compile the model with an optimizer (e.g., `Adam`), loss function (e.g., `SparseCategoricalCrossentropy`), and metrics (e.g., `accuracy`).
            *   Use `model.fit` to train the model, passing the training and validation data, epochs, and batch size.
        *   If using PyTorch:
            *   Define an optimizer (e.g., `Adam`) and loss function (e.g., `CrossEntropyLoss`).
            *   Iterate over the training data in batches, compute the loss, perform backpropagation, and update the model's weights.
            *   Evaluate the model on the validation data after each epoch.
*   **Code Cell:**
    *   Execution: Call `train_model()` to train the model.
*   **Markdown Cell:**
    *   Explanation: The model is now being trained on the synthetic data. Observe the training and validation loss to monitor the model's performance.

**Section 9: Visualizing Training Progress**

*   **Markdown Cell:**
    *   Title: "Visualizing Training Progress"
    *   Explanation:  Visualizing the training process can help us understand how well the model is learning.  We'll plot the training and validation loss over epochs to identify potential overfitting or underfitting. Overfitting occurs when the model learns the training data too well and performs poorly on unseen data. Underfitting occurs when the model fails to learn the underlying patterns in the data.
*   **Code Cell:**
    *   Function: `plot_loss_curves(history)`: Implement the function to plot the training and validation loss curves.  If using TensorFlow, the `history` object is returned by `model.fit`.  If using PyTorch, you'll need to manually store the loss values during training.
*   **Code Cell:**
    *   Execution: Call `plot_loss_curves()` to display the loss curves.
*   **Markdown Cell:**
    *   Explanation: This section displays the training and validation loss curves, providing insights into the model's learning progress.

**Section 10: Voice Synthesis**

*   **Markdown Cell:**
    *   Title: "Synthesizing Voice from Text"
    *   Explanation:  Now we'll use the trained model to synthesize voice from text.  This involves converting the text to phonemes, generating MFCCs using the model, and reconstructing an audio waveform from the MFCCs. We'll use a very simplified vocoder or directly generate random noise based on the model's output to demonstrate the process. We can use a simple one-to-one mapping here, so for each output label there is one audio to play.
*   **Code Cell:**
    *   Define a `phoneme_map`: a dictionary mapping each text label to a phoneme or audio snippet representation.
*   **Code Cell:**
    *   Function: `synthesize_voice(model, text, phoneme_map, sample_rate)`: Implement the function to synthesize voice from text.
        1.  Convert the `text` to a sequence of numerical labels based on the `phoneme_map`.
        2.  Generate MFCCs using the model.  (You may need to reshape the input to match the model's expected input shape.)
        3. Reconstruct audio by assigning each probability distribution to an audio data.
        4.  Return the synthesized audio as a numpy array.

*   **Code Cell:**
    *   Execution: Choose a text input. Call `synthesize_voice()` to generate the synthesized audio.  Use `IPython.display.Audio` to play the synthesized audio.
*   **Markdown Cell:**
    *   Explanation: This section demonstrates how to synthesize voice from text using the trained model and synthetic vocoder.  You should be able to hear a (potentially very noisy or distorted) synthesized voice.

**Section 11: Comparing Original and Synthesized Audio**

*   **Markdown Cell:**
    *   Title: "Comparing Original and Synthesized Audio"
    *   Explanation: This section allows you to listen to both the original synthetic audio and the audio generated from the model based on this section's synthetic label (text).
*   **Code Cell:**
    *   Execution: Plays the original audio generated and synthesized audio, allowing for a quick comparison.
*   **Markdown Cell:**
    *   Explanation: Compare the quality (or lack thereof) of the original and synthesized audio. Given the simplified approach and synthetic dataset, significant differences are expected, but this highlights the end-to-end process.

**Section 12: User Interaction**

*   **Markdown Cell:**
    *   Title: "User Interaction: Adjusting Parameters"
    *   Explanation: This section allows you to adjust parameters and rerun the analysis. We'll add sliders to control the number of epochs and the learning rate, so you can see how these parameters affect the training process and the quality of the synthesized voice. You can also adjust the input text and see the impact of this on the synthesized voice. Inline help will be added to describe each parameter.
*   **Code Cell:**
    *   Import `ipywidgets` (if not already imported).
*   **Code Cell:**
    *   Create `ipywidgets.IntSlider` widgets for `epochs` and create `ipywidgets.Text` widget for the `input_text`. Provide initial values, ranges, and descriptions for each slider.

*   **Code Cell:**
    *   Use `ipywidgets.interactive` to create an interactive function that takes the slider values and text as input and reruns the model training and voice synthesis steps.
*   **Markdown Cell:**
    *   Explanation: Now experiment with different parameters and observe how they impact the training process and the synthesized voice. For example, increasing the number of epochs might improve the model's performance but could also lead to overfitting.
**Section 13: References**

*   **Markdown Cell:**
    *   Title: "References"
    *   List any external datasets, libraries, or research papers used in the notebook. For example:
        *   `librosa`: [https://librosa.org/](https://librosa.org/)
        *   TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/) or PyTorch: [https://pytorch.org/](https://pytorch.org/)
        *   scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)

