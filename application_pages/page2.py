import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from application_pages.utils import create_model, train_model, plot_loss_curves_plotly, print_model_summary # Assuming utils.py exists

def run_page2():
    st.title("Model Training")

    st.markdown("""
    In this section, we move from data preparation to building and training our neural network model for voice synthesis. This model will learn to map the extracted MFCC features to their corresponding text labels.

    ### Data Splitting
    Before training, it's crucial to split our dataset into training and validation sets.
    *   **Training Set**: Used to train the model, allowing it to learn the patterns and relationships within the data.
    *   **Validation Set**: Used to evaluate the model's performance during training and to tune hyperparameters. It helps in detecting overfitting – when a model learns the training data too well but performs poorly on unseen data.

    We'll use an 80/20 split, meaning 80% of our data will be for training and 20% for validation, with a `random_state` for reproducibility.
    """)

    # Ensure audio_data and text_labels exist in session_state from Page 1
    # Try to restore from persisted keys if missing
    if 'audio_data' not in st.session_state or 'text_labels' not in st.session_state:
        st.warning("Please generate data on the 'Data Generation & Preprocessing' page and click 'Generate Data' before training.")
        return
    if 'mfccs_data' not in st.session_state:
        # Try to restore from persisted key
        if 'mfccs_data_persisted' in st.session_state:
            st.session_state['mfccs_data'] = st.session_state['mfccs_data_persisted']
        else:
            st.warning("Please preprocess audio on the 'Data Generation & Preprocessing' page before training.")
            return

    # --- Splitting Data ---
    st.header("Data Splitting")
    X = st.session_state['mfccs_data']
    y = st.session_state['text_labels']

    if X.shape[0] == 0 or y.shape[0] == 0:
        st.warning("No MFCCs or labels available to split. Please adjust data generation parameters on the previous page.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    st.write(f"Shape of `X_train`: `{X_train.shape}`")
    st.write(f"Shape of `X_val`: `{X_val.shape}`")
    st.write(f"Shape of `y_train`: `{y_train.shape}`")
    st.write(f"Shape of `y_val`: `{y_val.shape}`")

    st.markdown("""
    ### Neural Network Model Definition and Training
    Our simplified neural network model is designed to take the MFCC features as input and predict the corresponding text labels (phonemes in a more complex system). The model architecture consists of:

    *   **Convolutional Layers (`Conv2D`)**: These layers are excellent for extracting local patterns and features from the 2D MFCC input, similar to how they work with image data.
    *   **Max Pooling Layers (`MaxPooling2D`)**: These layers reduce the spatial dimensions of the feature maps, helping to make the model more robust to small variations in the input and reducing computational load.
    *   **Flatten Layer**: Transforms the 2D output of the convolutional layers into a 1D vector, preparing it for the dense layers.
    *   **Dense Layers**: Fully connected layers that learn complex non-linear relationships between the extracted features and the output labels.
    *   **Output Layer**: A final dense layer with a `softmax` activation function, which outputs a probability distribution over our `num_labels` possible text labels.

    We'll use **TensorFlow** to implement and train this model. The number of **epochs** (how many times the model sees the entire training dataset) is a crucial parameter you can adjust to see its impact on training.
    """)

    # --- Model Training ---
    st.header("Model Training")



    mfcc_shape = st.session_state['mfccs_data'].shape
    input_shape = mfcc_shape[1:] + (1,) # Add channel dimension for Conv2D
    num_labels = len(np.unique(st.session_state['text_labels']))

    # Check for zero frame dimension in MFCCs
    if mfcc_shape[1] == 0 or mfcc_shape[2] == 0:
        st.error(f"MFCC data has a zero dimension (shape: {mfcc_shape}), likely due to too short audio samples or preprocessing error. Please increase the 'Sample Length' in Data Generation & Preprocessing and click 'Generate Data' again before training.")
        return

    if num_labels == 0:
        st.warning("Cannot create model with zero labels. Please generate data with at least 1 unique label.")
        return

    # Create model if not in session state or if number of labels changed
    if 'model' not in st.session_state or \
       st.session_state.get('num_labels_model_prev') != num_labels:
        st.session_state['model'] = create_model(input_shape, num_labels)
        st.session_state['num_labels_model_prev'] = num_labels

    st.subheader("Model Summary")
    st.code(print_model_summary(st.session_state['model']), language='text')

    epochs = st.slider("Number of Epochs", min_value=1, max_value=20, value=10, help="Number of training epochs.", key="epochs_train")
    batch_size = 32 # Fixed batch size

    # Train model if not in session state or if epochs changed
    # Need to check if history exists AND if the number of epochs matches,
    # or if any data generation parameters changed (which would re-trigger model creation/data split)
    history_epochs = len(st.session_state['history'].history['loss']) if 'history' in st.session_state and hasattr(st.session_state['history'], 'history') and 'loss' in st.session_state['history'].history else 0

    if 'history' not in st.session_state or \
       history_epochs != epochs or \
       st.session_state.get('num_samples_prev') != st.session_state['num_samples_prev_for_training'] or \
       st.session_state.get('sample_length_prev') != st.session_state['sample_length_prev_for_training'] or \
       st.session_state.get('num_labels_gen_prev') != st.session_state['num_labels_gen_prev_for_training'] :
        
        st.session_state['history'] = train_model(st.session_state['model'], X_train, y_train, X_val, y_val, epochs, batch_size)
        # Store the parameters that led to this training, so we can re-train only when necessary
        st.session_state['num_samples_prev_for_training'] = st.session_state.get('num_samples_prev')
        st.session_state['sample_length_prev_for_training'] = st.session_state.get('sample_length_prev')
        st.session_state['num_labels_gen_prev_for_training'] = st.session_state.get('num_labels_gen_prev')
        
    st.success("Model training complete.")

    # --- Training Progress ---
    st.header("Visualizing Training Progress")
    st.markdown("""
    The **loss curve** is a fundamental diagnostic tool in machine learning. It plots the model's loss (error) on both the training and validation datasets over the course of training epochs.

    *   **Training Loss**: Indicates how well the model is learning from the data it has seen. A decreasing training loss suggests the model is improving its fit to the training data.
    *   **Validation Loss**: Indicates how well the model generalizes to new, unseen data. If the validation loss starts to increase while the training loss continues to decrease, it's a strong sign of **overfitting**.

    Monitoring these curves helps us understand if our model is learning effectively, if it's overfitting, or if it's underfitting (not learning enough).
    """)

    loss_fig = plot_loss_curves_plotly(st.session_state['history'])
    st.plotly_chart(loss_fig)
