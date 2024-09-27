#run streamlit run /Users/roshanrajendran/Desktop/ML_proj/driver_detection/app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('driver_action_model.h5')

# Define class names
class_names = ['Safe driving', 'Unsafe: Driver is texting', 'Unsafe: Driver is talking on the phone', 'Unsafe: Driver is texting',
               'Unsafe: Driver is on the phone', 'Unsafe: Driver is operating the radio', 'Unsafe: Driver is drinking', 'Unsafe: Driver is reaching behind',
               'Unsafe: Driver is distracted', 'Unsafe: Driver is talking to passenger']

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Predict the class of the uploaded image
def predict_class(img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    print(predictions)
    predicted_class = np.argmax(predictions)
    return class_names[predicted_class]

# Streamlit app
st.set_page_config(
    page_title="Autocom: A Driver Action Detection",
    page_icon="🚗",
    layout="centered"
)

st.markdown("<h2 style='text-align: left;'>AutoCom: your guardian on the road</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left;'>Welcome to AutoCom, your ultimate solution for enhancing driver safety and ensuring peace of mind on the road.</p>", unsafe_allow_html=True)

st.write("Upload an image to predict the driver's action.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        with st.spinner('Processing...'):
            label = predict_class(img)
        
        # Conditional formatting based on the prediction
        if label == 'Safe driving':
            st.success(f'Predicted action: {label}')
        else:
            st.error(f'Predicted action: {label}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image file.")
