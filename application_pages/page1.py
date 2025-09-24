import streamlit as st
import numpy as np
from application_pages.utils import create_synthetic_dataset, preprocess_audio, plot_spectrogram_plotly, plot_mfccs_plotly # Assuming utils.py exists

def run_page1():
    st.title("Data Generation & Preprocessing")

    st.markdown(r"""
    Welcome to the first stage of our Text-to-Speech lab: **Data Generation** and **Preprocessing**. Here, we'll create the foundational elements for our TTS system by generating synthetic audio data and transforming it into a format suitable for machine learning models.

    ### Synthetic Data Generation
    In a real-world scenario, collecting and labeling audio data can be a time-consuming and resource-intensive task. To simplify our learning process, we'll generate synthetic audio samples and assign them corresponding numerical text labels. This allows us to focus on the core concepts of TTS without dealing with large, complex datasets.

    Use the sliders below to control the characteristics of our synthetic dataset:
    """)


    # --- Data Generation ---
    st.header("Synthetic Data Generation")
    num_samples = st.slider("Number of Samples", min_value=10, max_value=200, value=100, help="The number of synthetic audio samples to generate.", key="num_samples")
    sample_length = st.slider("Sample Length", min_value=100, max_value=2000, value=1000, help="The number of data points for each audio sample.", key="sample_length")
    num_labels = st.slider("Number of Labels", min_value=2, max_value=20, value=10, help="The number of unique text labels.", key="num_labels_gen")

    generate = st.button("Generate Data", help="Click to generate and persist synthetic data.")

    # Only generate data when button is pressed, or if not present in session state
    if generate or 'audio_data' not in st.session_state:
        st.session_state['audio_data'], st.session_state['text_labels'] = create_synthetic_dataset(num_samples, sample_length, num_labels)
        st.session_state['num_samples_prev'] = num_samples
        st.session_state['sample_length_prev'] = sample_length
        st.session_state['num_labels_gen_prev'] = num_labels
        # Remove old MFCCs if present, force regeneration
        if 'mfccs_data' in st.session_state:
            del st.session_state['mfccs_data']
        if generate:
            st.rerun()


    st.write(f"Shape of synthetic audio data: `{st.session_state['audio_data'].shape}`")
    st.write(f"Shape of synthetic text labels: `{st.session_state['text_labels'].shape}`")
    st.audio(st.session_state['audio_data'][0], sample_rate=22050, format='audio/wav')
    st.caption("Example of a synthetic audio sample.")

    st.markdown("""
    ### Preprocessing: MFCC Extraction
    Raw audio waveforms are high-dimensional and often contain redundant information that can hinder a neural network's ability to learn effectively. **Feature extraction** is the process of transforming raw data into a set of more informative and manageable features.

    **Mel-Frequency Cepstral Coefficients (MFCCs)** are a widely used feature in speech processing. They represent the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear Mel scale of frequency. Essentially, MFCCs try to mimic how the human ear perceives sound.

    The general steps for MFCC extraction are:
    1.  **Framing**: Divide the audio signal into small, overlapping frames.
    2.  **Windowing**: Apply a window function (e.g., Hamming window) to each frame to minimize spectral leakage.
    3.  **Fast Fourier Transform (FFT)**: Compute the power spectrum of each frame.
    4.  **Mel Filter Bank**: Map the power spectrum onto the Mel scale using a filter bank. This scale is linear below 1 kHz and logarithmic above 1 kHz, reflecting human hearing.
    5.  **Log Energy**: Take the logarithm of the energy at each Mel frequency.
    6.  **Discrete Cosine Transform (DCT)**: Apply a DCT to the log Mel energies. The resulting coefficients are the MFCCs. Only the lower-order coefficients are typically kept as they represent the most perceptually relevant information.

    The formula for calculating MFCC's is given by:
    $$MFCC = DCT(\log(m))$$
    where $DCT$ stands for Discrete Cosine Transform and $m$ is the power spectrum of the audio after it passes through Mel filters.

    We will use the `librosa` library to extract MFCCs from our synthetic audio samples.
    """)

    # --- Preprocessing ---
    st.header("Preprocessing: MFCC Extraction")
    sample_rate = 22050
    st.write(f"Using a fixed sample rate of `{sample_rate} Hz` for preprocessing.")


    if 'mfccs_data' not in st.session_state or \
       st.session_state.get('num_samples_prev') != num_samples or \
       st.session_state.get('sample_length_prev') != sample_length or \
       st.session_state.get('num_labels_gen_prev') != num_labels:
        if st.session_state['audio_data'].shape[0] > 0 and st.session_state['audio_data'].shape[1] > 0:
            st.session_state['mfccs_data'] = preprocess_audio(st.session_state['audio_data'], sample_rate, sample_length=sample_length)
        else:
            st.session_state['mfccs_data'] = np.empty((0, 40, 0), dtype=np.float32)
        # Persist MFCCs for next page
        st.session_state['mfccs_data_persisted'] = st.session_state['mfccs_data']

    st.write(f"Shape of MFCCs data: `{st.session_state['mfccs_data'].shape}`")

    # --- Visualization ---
    st.header("Visualizing Audio Data and MFCCs")
    st.markdown("""
    Below, you can see a **spectrogram** of an original synthetic audio sample and its corresponding **MFCCs**. A spectrogram visually represents the frequencies of sound as they vary over time, while MFCCs provide a more compact and perceptually relevant representation.
    """)

    sample_index = st.slider("Select Sample Index for Visualization", 0, max(0, st.session_state['audio_data'].shape[0] - 1), value=0, key="viz_sample_index")

    if st.session_state['audio_data'].shape[0] > 0:
        sample_audio = st.session_state['audio_data'][sample_index, :]
        spectrogram_fig = plot_spectrogram_plotly(sample_audio, sample_rate, 'Spectrogram of Original Synthetic Audio')
        st.plotly_chart(spectrogram_fig)
    else:
        st.warning("No audio data available for spectrogram visualization.")

    if st.session_state['mfccs_data'].shape[0] > 0:
        sample_mfccs = st.session_state['mfccs_data'][sample_index, :, :]
        mfccs_fig = plot_mfccs_plotly(sample_mfccs, sample_rate, 'MFCCs of Synthetic Audio')
        st.plotly_chart(mfccs_fig)
    else:
        st.warning("No MFCC data available for visualization.")

    st.markdown("""
    By understanding these visualizations, we can appreciate how raw audio is transformed into features that a neural network can process for tasks like voice synthesis.
    """)
