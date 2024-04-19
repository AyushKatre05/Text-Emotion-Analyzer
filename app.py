import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import pickle

# Load the trained model
model = load_model("trained_model.h5")  # Replace "your_model.h5" with the path to your trained model

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open("label_encoder.pickle", "rb") as handle:
    label_encoder = pickle.load(handle)

# Load the maximum length
with open("max_length.pickle", "rb") as handle:
    max_length = pickle.load(handle)

# Define predict function
def predict_label(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    return predicted_label[0]

# Streamlit app
st.title("Emotion Detection Using Text")

input_text = st.text_input("Enter your text:")
if st.button("Predict"):
    if input_text:
        predicted_label = predict_label(input_text)
        st.write("Predicted Emotion:", predicted_label)
    else:
        st.write("Please enter some text.")
