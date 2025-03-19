import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# ✅ Define model path
MODEL_PATH = "PLANT-DISEASE-IDENTIFICATION/trained_plant_disease_model.keras"

# ✅ Load the model (only if the file exists)
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    st.error(f"Error: Model file not found at {MODEL_PATH}. Please check the file path!")

# ✅ Prediction function
def model_prediction(test_image):
    if model is None:
        return None  # Prevent prediction if model is missing
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# ✅ Sidebar
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ✅ Display logo/image
img_path = "PLANT-DISEASE-IDENTIFICATION/Diseases.png"
if os.path.exists(img_path):
    img = Image.open(img_path)
    st.image(img, use_container_width=True)
else:
    st.warning(f"Warning: Image file not found at {img_path}.")

# ✅ Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# ✅ Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        st.image(test_image, use_container_width=True)  # ✅ Fixed deprecation issue
    
        if st.button("Predict"):
            # st.snow()
            st.write("Our Prediction:")
            
            result_index = model_prediction(test_image)
            
            if result_index is None:
                st.error("Model not loaded. Unable to make a prediction.")
            else:
                # ✅ Class labels
                class_names = [
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
                
                disease_name = class_names[result_index]
                st.success(f"Model Prediction: {disease_name}")
