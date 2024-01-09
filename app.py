import streamlit as st
from PIL import Image
# from predict import predict_image
from ultralytics import YOLO
import numpy as np
from util import set_background


set_background('./bgs/background.jpg')

# Load YOLO model
model = YOLO('./runs/classify/train/weights/last.pt')

# Set Streamlit title
st.title('Weather Prediction')

# File uploader to get user input
uploaded_file = st.file_uploader("Choose an image...", type=['jpeg', 'jpg', 'png'])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Check if the 'Predict' button is clicked
    if st.button('Predict'):
        # Save the uploaded image to a temporary file
        temp_image_path = 'temp_image.jpg'
        image.save(temp_image_path)

        # Predict using the YOLO model
        results = model(temp_image_path)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        # Get the predicted class
        predicted_class = names_dict[np.argmax(probs)]

        # Display the prediction results
        st.write(
            f'<span style="font-size: 36px; font-weight: bold; color: #F3F2A7;">Prediction of Image : {predicted_class}</span>',
            unsafe_allow_html=True
        )

        st.write(f'Weather Probability of Image :')
        st.write(f'{names_dict}')
        st.write(f'{probs}')
        
        #print(names_dict)
        #print(probs)
