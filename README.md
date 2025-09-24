# QuLab: Custom Voice Creation with Streamlit

## Project Title and Description

QuLab is a Streamlit application that provides a hands-on introduction to custom voice creation using a simplified Text-to-Speech (TTS) system.  It demonstrates the core concepts of data generation, preprocessing (MFCC extraction), model training with a simple neural network, and voice synthesis. This lab allows users to experiment with parameters and observe their impact on the synthesized voice in real-time using synthetic data. This approach allows us to avoid the complexities and resource requirements involved in real-world audio processing and focuses on the core concepts.

## Features

*   **Synthetic Data Generation**: Generates synthetic audio data and corresponding text labels with adjustable parameters (number of samples, sample length, number of labels).
*   **MFCC Extraction**: Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the generated audio data using `librosa`.
*   **Spectrogram and MFCC Visualization**:  Displays interactive spectrograms and MFCC plots using `plotly` to visualize audio data and features.
*   **Neural Network Model Training**: Trains a simplified neural network model (built with `tensorflow.keras`) to map MFCC features to text labels.
*   **Loss Curve Visualization**: Plots training and validation loss curves using `plotly` to monitor model training progress.
*   **Voice Synthesis**: Synthesizes voice from input text labels using the trained model and a simplified mapping to predefined audio snippets.
*   **Interactive Parameter Adjustment**: Allows users to adjust training epochs and input labels, retraining the model and synthesizing new audio in real-time.
*   **Comparison of Original and Synthesized Audio**: Allows comparing original synthetic audio snippets with the model's synthesized output to evaluate performance.
*   **Clear Documentation**: Provides explanations and interactive elements to guide users through the process.

## Getting Started

### Prerequisites

*   Python 3.7 or higher
*   Pip package installer
*   Basic understanding of audio processing and machine learning concepts (optional)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    If a `requirements.txt` file isn't provided, manually install the necessary libraries:

    ```bash
    pip install streamlit numpy pandas scipy librosa tensorflow scikit-learn plotly
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application in your web browser:**

    Streamlit will automatically open the application in your default web browser.  If it doesn't, look for the URL printed in the terminal (usually `http://localhost:8501`).

3.  **Navigate through the application:**

    *   Use the sidebar to select different sections: "Data Generation & Preprocessing," "Model Training," and "Voice Synthesis & Interaction."
    *   Follow the instructions and interact with the sliders, number inputs, and buttons on each page.
    *   Observe the generated data, plots, and synthesized audio to understand the different stages of the TTS system.

## Project Structure

```
QuLab/
├── app.py                       # Main Streamlit application script
├── application_pages/           # Directory containing individual page scripts
│   ├── page1.py               # Data Generation & Preprocessing
│   ├── page2.py               # Model Training
│   ├── page3.py               # Voice Synthesis & Interaction
│   └── utils.py               # Utility functions (data generation, preprocessing, model definition, etc.)
├── README.md                    # This file
├── requirements.txt             # (Optional) List of Python dependencies
└── ...
```

*   `app.py`: The entry point for the Streamlit application.  It defines the main layout and navigation.
*   `application_pages/`: This directory organizes the code for each of the application's main sections.
    *   `page1.py`:  Handles synthetic data generation, MFCC extraction, and visualization of the spectrogram and MFCCs.
    *   `page2.py`:  Deals with data splitting, neural network model creation and training, and loss curve visualization.
    *   `page3.py`:  Implements voice synthesis, comparison of original and synthesized audio, and interactive parameter adjustment.
    *   `utils.py`:  Contains reusable utility functions for data generation, audio preprocessing, model creation, training, synthesis, and plotting.
*   `README.md`:  Provides documentation and instructions for the project.
*   `requirements.txt`: (Optional) A text file listing the Python packages required to run the application.

## Technology Stack

*   **Streamlit:**  For building the interactive web application.
*   **NumPy:** For numerical operations and array manipulation.
*   **Pandas:** For data manipulation and analysis (potentially unused, but imported).
*   **SciPy:** For scientific computing (potentially unused, but imported).
*   **Librosa:** For audio analysis and feature extraction (MFCCs).
*   **TensorFlow/Keras:** For building and training the neural network model.
*   **Scikit-learn:** For data splitting (train/test split).
*   **Plotly:** For creating interactive plots and visualizations.

## Contributing

(Optional)

We welcome contributions to QuLab! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request to the main branch.

Please follow our coding style guidelines and include relevant tests with your contributions.

## License

(Specify the License - e.g., MIT License)

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.

## Contact

[Your Name/Organization]

[Your Email/Website/Repository Link]
