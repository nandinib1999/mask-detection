import tensorflow as tf 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 128

# Head of the base model
base = vgg16.VGG16(weights="imagenet", include_top=False, pooling='max', classes=2, input_tensor=Input(shape=(128,128,3)))

# The Fully Connected Model that will be trained 
head = base.output
head = Flatten(name="flatten")(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(1, activation="sigmoid")(head)

# Combining the head and FC model for training
model = Model(inputs=base.input, outputs=head)

# all the layers of the base model i.e. VGG16 are freezed so that the weights of VGG16 layers are not updated
for layer in base.layers:
	layer.trainable = False

# A callable function to return the model created
def create_model():
	return model