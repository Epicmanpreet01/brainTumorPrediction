import streamlit as st
import os
from PIL import Image

def augment_img(img_path):
    img = Image.open(img_path)
    return img.resize((1920,1080))

def show_eda():
    st.title("Data Analysis: Brain Tumor MRI")

    st.subheader("Dataset Distribution")
    st.image(augment_img("images/train_set_dist.png"), caption="Training Set Class Distribution")
    st.image(augment_img("images/test_set_dist.png"), caption="Test Set Class Distribution")

    st.subheader("Random Sampling from Training Data")
    st.image(augment_img("images/random_sampling.png"), caption="Sample MRI Images")

    st.subheader("Model Training History")
    st.image(augment_img("images/model_training_history.png"), caption="Training Accuracy & Loss")

    st.subheader("Confusion Matrix")
    st.image(augment_img("images/confusion_matrix.png"), caption="Confusion Matrix")

    st.subheader("ROC Curve")
    st.image(augment_img("images/roc_auc_curve.png"), caption="ROC Curve for Each Class")

    st.subheader("Sample Predictions on Test Images")
    cols = st.columns(4)
    class_images = {
        "glioma": "predict_glioma.png",
        "meningioma": "predict_meningioma.png",
        "notumor": "predict_notumor.png",
        "pituitary": "predict_pituitary.png"
    }
    for i, (label, img_name) in enumerate(class_images.items()):
        with cols[i]:
            st.image(f"images/{img_name}", caption=f"Prediction: {label.capitalize()}", use_column_width=True)
