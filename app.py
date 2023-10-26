import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model (assuming it's in the same directory)
model = load_model('model1.hdf5')

# Define the classes
class_names = ['Real', 'Fake']

# Create a function to make predictions
def predict(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Resize the image to match model input size
        img = np.array(img)
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        return class_names[int(np.round(prediction[0][0]))], prediction[0][0]
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

# Streamlit app
st.set_page_config(page_title="Real vs. Fake Image Classifier")

# Custom CSS
st.markdown(
    """
    <!-- Your CSS styles here -->
    """,
    unsafe_allow_html=True,
)

# Logo image above the title
logo_image = Image.open('image1.png')  # Use a relative path
st.image(logo_image, use_column_width=True)

st.title("Real vs. Fake Image Classifier")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    prediction, confidence = predict(uploaded_image)
    if prediction is not None:
        st.markdown(f"<div class='stMarkdown'>{prediction} Image</div>", unsafe_allow_html=True)

        # Display prediction results with probabilities
        st.markdown("<div class='prediction-results'>", unsafe_allow_html=True)

        if prediction == 'Real':
            st.markdown("<div class='prediction-label'>REAL</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='probability'>{confidence:.2f} %</div>", unsafe_allow_html=True)
            st.progress(confidence / 100)  # Convert to [0, 1] range
        else:
            st.markdown("<div class 'prediction-label'>FAKE</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='probability'>{(100 - confidence):.2f} %</div>", unsafe_allow_html=True)
            st.progress((100 - confidence) / 100)  # Convert to [0, 1] range

        st.markdown("</div>", unsafe_allow_html=True)
