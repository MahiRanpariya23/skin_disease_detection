import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ------------------ LOAD MODEL ------------------
model = load_model("skin_model.h5")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Skin Disease Detection", layout="wide")

# ------------------ SIDEBAR ------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.info("Skin Disease Detection using AI")

# ------------------ HOME ------------------
if page == "🏠 Home":
    st.title("🩺 Skin Disease Detection System")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 👋 Welcome
        
        This project uses Deep Learning (MobileNetV2) to detect skin diseases from images.
        
        ### 🔍 Features:
        - Upload skin image
        - AI-based prediction
        - Risk level detection
        - Recommendation system
        
        ### 🧠 How it works:
        Image → Model → Prediction → Result
        
        ---
        ### 👉 Steps:
        1. Go to Predict page
        2. Upload image
        3. View result
        """)

    with col2:
        st.success("✔️ AI Powered")
        st.success("✔️ Fast Prediction")
        st.success("✔️ Easy to Use")

# ------------------ PREDICT ------------------
elif page == "🔍 Predict":

    st.title("🔍 Skin Disease Prediction")

    # Disease Info
    disease_info = {
        0: ("Actinic Keratoses", "Rough, scaly patches caused by sun damage."),
        1: ("Basal Cell Carcinoma", "Most common skin cancer."),
        2: ("Benign Keratosis", "Non-cancerous growth."),
        3: ("Dermatofibroma", "Small firm bump."),
        4: ("Melanoma", "Dangerous skin cancer."),
        5: ("Melanocytic Nevus", "Common mole."),
        6: ("Vascular Lesion", "Blood vessel abnormality")
    }

    # Risk Levels
    risk_levels = {
        0: "Medium Risk ⚠️",
        1: "High Risk 🚨",
        2: "Low Risk ✅",
        3: "Low Risk ✅",
        4: "High Risk 🚨",
        5: "Low Risk ✅",
        6: "Medium Risk ⚠️"
    }

    # ------------------ FILE UPLOADER ------------------
    uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg", "png", "jpeg"])

    # ------------------ PREDICTION ------------------
    if uploaded_file:

        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=300)

        with col2:
            # Preprocess
            img = img.resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0]
            index = np.argmax(prediction)
            confidence = prediction[index] * 100

            disease_name, description = disease_info[index]
            risk = risk_levels[index]

            # ------------------ RESULT ------------------
            st.subheader("📊 Result")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("### Disease")
                st.write(disease_name)

            with colB:
                st.markdown("### Confidence")
                st.write(f"{confidence:.2f}%")

            # Progress bar
            st.progress(int(confidence))

            # Risk level
            if "High" in risk:
                st.error(f"Risk Level: {risk}")
            elif "Medium" in risk:
                st.warning(f"Risk Level: {risk}")
            else:
                st.success(f"Risk Level: {risk}")

        # ------------------ EXTRA INFO ------------------
        st.markdown("---")

        st.subheader("📖 Description")
        st.write(description)

        st.subheader("💡 Recommendation")

        if "High" in risk:
            st.error("Immediate consultation with a dermatologist is recommended.")
        elif "Medium" in risk:
            st.warning("Monitor the condition and consider medical advice.")
        else:
            st.success("Condition appears harmless. Monitor regularly.")

        st.warning("⚠️ Not a medical diagnosis.")

    else:
        st.info("Upload an image to start prediction.")

# ------------------ ABOUT ------------------
elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
    ### 📌 Project Details

    Skin disease detection using Deep Learning.

    ### 🧠 Model:
    - MobileNetV2 (Transfer Learning)
    - HAM10000 Dataset

    ### 🛠️ Tech:
    - Python
    - TensorFlow / Keras
    - Streamlit

    ### 🎯 Goal:
    Help users identify skin conditions using AI.
    """)