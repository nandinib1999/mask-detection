# Medical Mask Detection
This project can be used to detect the medical mask on a person's face. It sends an alert to the console in case mask is not detected. The project aims to help the GOI to make the security camera smart with the mask detection. It can prove to be very useful in today's time when wearing masks is more about necessity than a fashion due to the deadly COVID 19 pandemic.

## Workflow
1. The first step is to detect the faces in the video or webcam live feed. 
2. The detected face is then sent to the classification model to predict whether the mask is present or not.

## Dataset
1. For the face detection, I have used a pre-trained face detector which can detect the faces in various different poses. The haarcascade_frontalface_alt.xml was not working efficiently to detect the face when it was covered with a mask.
2. For the mask covered face detection, I used the dataset available at [Link](https://github.com/prajnasb/observations/tree/master/experiements/data), [Medical Mask Dataset](https://www.kaggle.com/vtech6/medical-masks-dataset) and downloaded some images from Google. 

Currently, I have 1672 images in total for both masked and non-masked faces. The model accuracy can be increased significantly by adding more variety of images to the dataset for training.

## Usage

- **preprocess_dataset.py** - This file can be used to split the images stored in dataset folder 

- **model.py** - In this file, the model used for the classification task has been defined. I have extended the VGG16 model for the classification purposes.

- **train.py** - The training of model can be performed using this python script. Currently, the model trains for 50 epochs while monitoing the validation loss for early stopping. When training is complete, the model is saved in h5 format for later use.

- **run.py** - This file can be used to test the trained model. It supports two kinds of input - webcam feed or a video input. The option to be selected can be reflected with cmd arguments --play_video and --webcam. By default, both the arguments are set to False.

