import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from PIL import Image
import streamlit as st

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model.keras")

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(uploaded_file):
    try:
        model = load_model()
        img = Image.open(uploaded_file).convert("RGB").resize((128,128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        return class_labels[class_idx], confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Error", 0.0