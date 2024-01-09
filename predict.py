from ultralytics import YOLO

import numpy as np


model = YOLO('./runs/classify/train/weights/last.pt')  # load a custom model

results = model('C:/Users/Priyangshu/Object_Detection_computer_Vision/dataset/alpaca/2b9f929ad756b54e.jpg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])