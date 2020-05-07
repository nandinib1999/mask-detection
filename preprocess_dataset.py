import os
import xmltodict
from tqdm import tqdm
import cv2
import random

'''
Found 1341 images belonging to 2 classes.
Found 331 images belonging to 2 classes.
'''

######## GLOBAL VARIABLES #########

IMAGE_DIR = 'medical-masks-dataset/images'
XML_DIR = 'medical-masks-dataset/labels'

TRAIN_DIR = "train/"
TEST_DIR = "test/"
MODEL_DIR = "model/"
test_split = 0.2

def create_directory(path):
	if not os.path.exists(path):
		os.mkdir(path)

def create_label_directory():
	print('Creating Directories ... ')
	create_directory(TRAIN_DIR)
	create_directory(MODEL_DIR)
	create_directory(TEST_DIR)
	create_directory(os.path.join(TRAIN_DIR, 'with_mask')) # With Mask
	create_directory(os.path.join(TRAIN_DIR, 'without_mask')) # Without Mask
	create_directory(os.path.join(TEST_DIR, 'with_mask')) # With Mask
	create_directory(os.path.join(TEST_DIR, 'without_mask')) # Without Mask

def save_images(cropped_img_list, test_cnt):
	cnt = 0
	for cropped_img in tqdm(cropped_img_list):
		cnt += 1
		img, label = cropped_img
		if cnt <= test_cnt:
			filename = 'test_'+str(cnt)+'.jpg'
			folder = TEST_DIR
		else:
			filename = 'train_'+str(cnt)+'.jpg'
			folder = TRAIN_DIR
		cv2.imwrite(os.path.join(folder, label, filename), img)

def perform_preprocessing():
	create_label_directory()
	all_cropped_images = []
	none_cnt = 0
	for folder in os.listdir('dataset'):
		for file in os.listdir(os.path.join('dataset', folder)):
			image_path = os.path.join('dataset', folder, file)
			img = cv2.imread(image_path)
			img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
			all_cropped_images.append([img, folder])

	# Performing shuffle over the images
	random.shuffle(all_cropped_images)
	test_dataset_cnt = int(len(all_cropped_images) * test_split)
	print("Total Dataset: ", len(all_cropped_images))
	print("Test Dataset: ", test_dataset_cnt)
	print("Train Dataset: ", len(all_cropped_images) - test_dataset_cnt)

	print('Saving Images after split..')
	save_images(all_cropped_images, test_dataset_cnt)
	print("Total Dataset: ", len(all_cropped_images))

if __name__ == '__main__':
	perform_preprocessing()