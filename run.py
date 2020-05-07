import tensorflow as tf
from tensorflow.keras.models import load_model

import cv2
from PIL import Image
import dlib

import numpy as np
import argparse
import os

##### GLOBALS #####
font_scale=1
thickness = 2
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
font=cv2.FONT_HERSHEY_SIMPLEX

######### Commandline Arguments #########
argument = argparse.ArgumentParser()
argument.add_argument("-m", "--model", help="path to the model to be used", default="model_vgg5.h5")

argument.add_argument("-v", "--play_video", help="True/False - play a mp4 video", default=False)
argument.add_argument("-vpath", "--video_path", help="path to the mp4 video to be played", default="webcam/video5.mp4")

argument.add_argument("-w", "--webcam", help="True/False - start webcam", default=False)
args = argument.parse_args()

# Loading the trained model for medical mask detection
model_filename = str(args.model)
print("Loading mask detector..")
model = load_model(model_filename)

# Loading the face detector 
print("Loading face detector...")
prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector",
    "res10_300x300_ssd_iter_140000.caffemodel")
face_model = cv2.dnn.readNet(prototxtPath, weightsPath)

# Select whether to open webcam or a video
if args.webcam:
    cap = cv2.VideoCapture(0)
elif args.play_video:
    cap = cv2.VideoCapture(args.video_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (128, 128),
            (104.0, 177.0, 123.0))
        face_model.setInput(blob)
        detections = face_model.forward()
        faces = []
        bndbox = []
        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # This is to ensure that the bounding box is not out of the frame.
                (x1, y1) = (max(0, x1), max(0, y1))
                (x2, y2) = (min(w - 1, x2), min(h - 1, y2))

                # Crop the face (ROI) from the frame
                face = frame[y1:y2, x1:x2]
                # Performing preprocessing 
                face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
                face = tf.reshape(face, (1, 128, 128, 3))
                face = tf.cast(face, dtype=tf.float32)
                # Denormalizing the image
                face = face - [123.68, 116.779, 103.939]
                pred = model.predict(face)
                prediction = np.argmax(pred, axis=1)

                if prediction == 0:
                    cv2.putText(frame, "Masked", (x1,y1 - 10), font, font_scale, green, thickness)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)
                elif prediction == 1:
                    print('ALERTTTTTTTT!!!!!! NO MASK DETECTED !!!!!!!')
                    cv2.putText(frame, "No Mask", (x1,y1 - 10), font, font_scale, red, thickness)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2)
        
        cv2.imshow('frame',frame)
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
