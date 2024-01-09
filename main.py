from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='C:/Users/Priyangshu/ImageClassification(ComputerVision)/image_classification_with_YOLOV8/weather_dataset', epochs=20, imgsz=64)