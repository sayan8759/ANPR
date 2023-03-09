import cv2
from deeplearning import yolo_predictions, net
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
import json
from skimage import io
import re
pt.pytesseract.tesseract_cmd = r"C:\Program Files\tesseract\build\bin\Debug\tesseract.exe"


# Initialize video capture object with default camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

output_list = []
text_file = open("text_file.txt", "w", encoding = 'utf-8') # Open a file for writing text

while cap.isOpened():
    ret, frame = cap.read()

    if ret == False:
        print('Unable to read video')
        break

    results = yolo_predictions(frame, net)
    results_list = list(results)
    # Append the predictions to the output_list
    output_list.append(results_list)

    cv2.namedWindow('YOLO',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('YOLO',results)
    if cv2.waitKey(30) == 27 :
        break
    
    # Assume you have already extracted the text from the bounding boxes using Tesseract OCR
    text = pt.image_to_string(frame)
    print(text)
    text_file.write(text + "\n") # Write the text to the file

# Write the output_list to a JSON file


# Close the text file and release video capture object
text_file.close()
cap.release()
cv2.destroyAllWindows()
