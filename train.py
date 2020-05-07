import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model
import matplotlib.pyplot as plt 
import argparse

####### GLOBALS #########
batch_size = 16
TRAIN_DIR = "train/"
TEST_DIR = "test/"

######### Commandline Arguments #########
argument = argparse.ArgumentParser()
argument.add_argument("-e", "--epochs", help="number of epochs", default=50)
argument.add_argument("-m", "--model_name", help="saved model file name", default="model.h5")
argument.add_argument("-p", "--show_plots", help="show the accuracy and loss plots")
args = argument.parse_args()


# Creating pipelines for training and testing pipeline with image augmentation functions
train_data = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.4, height_shift_range=0.4, brightness_range=(0.67, 1.0), shear_range=0.3, zoom_range=0.3, horizontal_flip=True, fill_mode='nearest').flow_from_directory(batch_size=batch_size, directory=TRAIN_DIR, shuffle=True, target_size=(128,128), class_mode="binary")

validation_data = ImageDataGenerator(rescale=1./255).flow_from_directory(batch_size=batch_size, directory=TEST_DIR, shuffle=True, target_size=(128,128), class_mode="binary")

# Create Model using model.py
model = model.create_model()
print(model.summary())

epochs = int(args.epochs)

# Compiling the model
model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy']
)

# Fitting the model on training dataset
history = model.fit_generator(
	train_data,
	epochs = epochs,
	steps_per_epoch=len(train_data),
	validation_data=validation_data,
	validation_steps=len(validation_data)
)

if args.show_plots:
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()


	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

# Saving the trained model
model.save(args.model_name)