import streamlit as st
from streamlit_option_menu import option_menu
from utils.model_utils import predict_image
from utils.eda_utils import show_eda, augment_img

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Brain Tumor App",
        ["Predict", "Data Analysis"],
        icons=["upload", "bar-chart"],
        menu_icon="brain", default_index=0
    )

if selected == "Predict":
    st.title("Brain Tumor MRI Classification")
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        label, confidence = predict_image(uploaded_file)
        if label == "notumor":
            st.success(f"No Tumor Detected (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"Tumor Detected: `{label}` (Confidence: {confidence:.2f}%)")
        st.image(augment_img(uploaded_file), caption="Uploaded Image", use_column_width=True, width=200)

elif selected == "Data Analysis":
    show_eda()  