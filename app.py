import streamlit as st
from PIL import Image
import os

from src.predict import predict_demand

st.title("📊 Demand Prediction Engine")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=500)

    temp_path = "temp.jpg"
    image.save(temp_path)

    st.write("🔍 Predicting demand...")

    prediction = predict_demand(temp_path)

    if prediction is not None:

        # 🔥 Category logic
        if prediction < 10:
            category = "Low Demand"
        elif prediction < 30:
            category = "Medium Demand"
        else:
            category = "High Demand"

        st.success(f"📈 Predicted Demand: {prediction:.2f} units")
        st.info(f"📊 Category: {category}")

    else:
        st.error("Error processing image")

    os.remove(temp_path)