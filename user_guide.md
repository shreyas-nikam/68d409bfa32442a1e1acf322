id: 68d409bfa32442a1e1acf322_user_guide
summary: study on different Text-to-Speech technologies User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Custom Voice Creation with Synthetic Data: A Codelab

This codelab guides you through building a simplified Text-to-Speech (TTS) system using synthetic data. This approach allows us to focus on the core concepts without the complexities of real-world audio. You'll learn about data generation, preprocessing with Mel-Frequency Cepstral Coefficients (MFCCs), model training, and voice synthesis. The interactive nature of this application lets you explore how different parameters affect the final output.

## Data Generation & Preprocessing
Duration: 0:15:00

This step focuses on creating synthetic audio data and preparing it for model training.

### Synthetic Data Generation
We'll generate synthetic audio samples and assign them numerical text labels. You can control the dataset's characteristics using sliders:

*   **Number of Samples:**  Determines the size of the dataset. More samples generally lead to better model training, but also increase computational time.
*   **Sample Length:** Represents the duration of each audio sample.
*   **Number of Labels:** Sets the number of unique text labels the model will learn to associate with audio snippets.

The application will display the shape of the generated audio data and text labels. You can even listen to an example of a synthetic audio sample.

<aside class="positive">
<b>Tip:</b> Experiment with different values for these parameters to see how they affect the generated data. A larger number of samples might result in better model training, but also increases the processing time.
</aside>

### Preprocessing: MFCC Extraction
Raw audio waveforms are high-dimensional and often contain redundant information.  Therefore, we need to convert them into a more manageable format using **Mel-Frequency Cepstral Coefficients (MFCCs)**.

MFCCs represent the short-term power spectrum of a sound, mimicking how the human ear perceives sound. The process involves framing the audio, applying a window function, performing a Fast Fourier Transform (FFT), using a Mel filter bank, taking the logarithm of the energy, and applying a Discrete Cosine Transform (DCT).

The formula for calculating MFCC's is given by:
$$MFCC = DCT(\log(m))$$
where $DCT$ stands for Discrete Cosine Transform and $m$ is the power spectrum of the audio after it passes through Mel filters.

The application uses a fixed sample rate of 22050 Hz for preprocessing and displays the shape of the resulting MFCCs data.

### Visualizing Audio Data and MFCCs

A **spectrogram** visually represents the frequencies of sound as they vary over time.  **MFCCs** provide a more compact and perceptually relevant representation. You can select a sample index to visualize its spectrogram and MFCCs. This helps you understand how raw audio is transformed into features suitable for machine learning.

<aside class="negative">
<b>Warning:</b> If you don't see any data, make sure you've generated data in the "Synthetic Data Generation" section.
</aside>

## Model Training
Duration: 0:10:00

In this step, we'll build and train a neural network model to map MFCC features to text labels.

### Data Splitting
We split the dataset into training and validation sets. The training set is used to train the model, while the validation set helps evaluate the model's performance and prevent overfitting. An 80/20 split is used, meaning 80% of the data will be for training and 20% for validation. The application displays the shape of each set.

### Neural Network Model Definition and Training
The neural network model consists of Convolutional Layers (`Conv2D`), Max Pooling Layers (`MaxPooling2D`), a Flatten Layer, Dense Layers, and an Output Layer with a `softmax` activation function. We use TensorFlow to implement and train this model.

You can adjust the **number of epochs** to see its impact on training. The model summary is displayed, showing the architecture of the neural network.

<aside class="positive">
<b>Tip:</b> Experiment with different epoch values. More epochs can lead to better training, but also increase the risk of overfitting.
</aside>

### Visualizing Training Progress
The **loss curve** plots the model's loss (error) on the training and validation datasets over epochs. A decreasing training loss indicates the model is learning, while the validation loss shows how well the model generalizes to unseen data. An increasing validation loss, while the training loss decreases, signifies overfitting.

## Voice Synthesis & Interaction
Duration: 0:15:00

This step focuses on generating audio from text inputs using the trained model and allows for interactive experimentation.

### Voice Synthesis from Trained Model
For each input 'text' (represented by a numerical label), the model generates a corresponding audio output.  For demonstration, the application utilizes a one-to-one mapping: for each predicted label, a pre-defined synthetic audio snippet will be played. The spectrogram of the synthesized audio is displayed.

<aside class="negative">
<b>Warning:</b> If you encounter errors, ensure you have completed the previous steps and that the input label is within the valid range.
</aside>

### Comparing Original and Synthesized Audio

You can compare an original synthetic audio snippet with the audio synthesized by the model for the same label. This helps assess how well the model reproduces the target sound. Both the original and synthesized audio, along with their spectrograms, are displayed for comparison.

### Interactive Parameter Adjustment

This section allows you to experiment with training parameters and observe their real-time impact.  You can adjust the **epochs for interactive training** and specify multiple labels for synthesis. After clicking "Run Interactive Synthesis & Retrain", the model is retrained with the new epoch value, and the synthesized audio for the specified labels is generated.

A new interactive training loss curve is plotted after re-training for the interactive session.
<aside class="positive">
<b>Tip:</b> Use the interactive mode to see how different epochs values influence training and voice synthesis.
</aside>
