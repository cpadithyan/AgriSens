import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Function to load and predict using the trained model
def model_prediction(test_image):
    model_path = "trained_plant_disease_model.keras"

    # Check if the model file exists
    if not os.path.exists(model_path):
        st.error("Error: Model file not found. Please check the file path.")
        return None

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch

    # Make predictions
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of the max element

# Sidebar
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display logo/image
if os.path.exists("PLANT-DISEASE-IDENTIFICATION/Diseases.png"):
    img = Image.open("PLANT-DISEASE-IDENTIFICATION/Diseases.png")
    st.image(img, use_container_width=True)
else:
    st.warning("Warning: Diseases.png file not found!")

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    # Upload Image
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")

            result_index = model_prediction(test_image)

            if result_index is not None:
                # Class labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                st.success(f"Model is predicting it's a **{class_name[result_index]}**")

