import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the pre-trained model and tokenizer
model = load_model('model/spam_classifier_model.h5')
with open('model/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load training history if available
try:
    with open('model/history.pkl', 'rb') as file:
        history = pickle.load(file)  # History object from training
except FileNotFoundError:
    history = None

# Preprocess function to clean input text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.strip()  # Remove extra whitespace
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Function to classify a single message
def classify_message(message):
    try:
        processed_input = preprocess_text(message)
        print(f"Processed Input: {processed_input}")  # Debugging line
        input_seq = tokenizer.texts_to_sequences([processed_input])
        print(f"Tokenized Sequence: {input_seq}")  # Debugging line
        input_pad = pad_sequences(input_seq, maxlen=50, padding='post')
        print(f"Padded Sequence: {input_pad}")  # Debugging line
        predicted_class = (model.predict(input_pad) > 0.5).astype("int32")
        return "Spam" if predicted_class[0][0] == 1 else "Ham"
    except Exception as e:
        print(f"Error processing message: {message}")
        print(f"Error details: {e}")
        return "Error"

# Corrected Function to extract model architecture
def get_model_architecture():
    architecture = []
    for layer in model.layers:
        layer_info = {
            "Layer Name": layer.name,
            "Layer Type": type(layer).__name__,
        }
        # Extract input and output shapes safely
        layer_info["Input Shape"] = getattr(layer, "input_shape", "N/A")
        layer_info["Output Shape"] = getattr(layer, "output_shape", "N/A")
        # Include configuration details
        layer_info["Config"] = layer.get_config() if hasattr(layer, "get_config") else "N/A"
        architecture.append(layer_info)
    return architecture

# App layout
st.title("📨 Spam or Ham Classifier")
st.write("Classify messages as **Spam** or **Ham**. Explore model architecture and details.")

# Tabs for functionality
tab1, tab3 = st.tabs(["📄 Single Prediction", "📊 Model Details"])

# Single Prediction Tab
with tab1:
    st.header("📄 Single Message Prediction")
    user_input = st.text_area("Enter your message:")
    if st.button("Classify Message"):
        if user_input:
            prediction = classify_message(user_input)
            st.success(f"Prediction: **{prediction}**")
        else:
            st.error("Please enter a message to classify.")


# Model Details Tab
with tab3:
    st.header("📊 Model Details")
    st.subheader("Model Summary")
    with st.expander("View Model Summary"):
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        st.text("\n".join(summary_str))
    
    st.subheader("Layer Details")
    architecture = get_model_architecture()
    for layer in architecture:
        st.write(layer)

    st.subheader("Layer Weights and Biases")
    for layer in model.layers:
        if hasattr(layer, "get_weights"):
            weights = layer.get_weights()
            if len(weights) == 2:  # Weights and Biases
                st.text(f"Layer: {layer.name}")
                st.text(f"Weights Shape: {np.shape(weights[0])}")
                st.text(f"Biases Shape: {np.shape(weights[1])}")
            elif len(weights) == 1:  # Only weights, no biases
                st.text(f"Layer: {layer.name}")
                st.text(f"Weights Shape: {np.shape(weights[0])}")
                st.text("No Biases for this layer.")
            else:
                st.text(f"Layer: {layer.name} has no weights or biases.")
        else:
            st.text(f"Layer: {layer.name} has no weights or biases.")


# Metrics Tab
# Footer
st.markdown("---")
st.markdown("Developed using Keras and Streamlit .")
