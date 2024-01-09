# Weather Detection Image Classifier with YOLOv8

This project utilizes YOLOv8, a state-of-the-art real-time object detection system, to create a weather detection image classifier. It is built with the Streamlit framework for a user-friendly interface.

## Overview

The application allows users to upload an image, and the YOLOv8 model predicts the weather conditions present in the image. The predicted class and probabilities are displayed along with the uploaded image.

## Requirements


```bash
pip install -r requirements.txt  ## To ensure that you have the required dependencies installed.
pip install git+https://github.com/ultralytics/yolov5  # Install YOLOv8


streamlit run app.py  # To Run Web API in your local System