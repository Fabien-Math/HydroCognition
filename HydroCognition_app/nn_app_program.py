## MODULES
import os
import cv2
import tensorflow as tf
import numpy as np

def load_validation_data(path):
	valid_images = []
	valid_labels = []
	valid_boxes = []

	with open(path + "/valid/data.csv", 'r') as data_file:
		data_file.readline()        # Pass header line
		for line in data_file:
			line = line.strip().split(',')
			image_path = line[0]
			valid_images.append(cv2.imread(path + "/valid/" + image_path))
			valid_labels.append([int(line[3])])
			valid_boxes.append([float(line[4]), float(line[5]), float(line[6]), float(line[7])])

	return np.array(valid_images), np.array(valid_labels), np.array(valid_boxes)

def process_dataset(images, labels, boxes, width, height):
	## Normalize pixel values to be between 0 and 1
	images = images / 255.0

	## Normalize boxes position to be between 0 and 1 for height and width
	boxes[:,0], boxes[:,2] = boxes[:,0]/width, boxes[:,2]/width
	boxes[:,1], boxes[:,3] = boxes[:,1]/height, boxes[:,3]/height

	## Shuffling all the data
	seed = 1000
	images = tf.random.shuffle(images, seed=seed)
	labels = tf.random.shuffle(labels, seed=seed)
	boxes = tf.random.shuffle(boxes, seed=seed)

	## Create validations sets
	verif = (labels, boxes)

	return images, verif

def load_model(model_name = "RCNN_DS_BS64_Fmax_E50_N64_NL2", path = "E:/Documents/Cours/SeaTech/2A/Git/App"):
	try:
		model = tf.keras.models.load_model(path + '/' + model_name + ".keras")
	except ValueError:
		return -1
	return model

def evaluate_model(model, width = 256.0, height = 144.0, path = "E:/Documents/Cours/SeaTech/2A/Projet/fishDataset2"):
	## Loading validation data
	valid_images, valid_labels, valid_boxes = load_validation_data(path)

	## Validation set informations
	nb_images = len(valid_images)

	## Processing validation data
	valid_images, valid_true = process_dataset(valid_images, valid_labels, valid_boxes, width, height)

	## Evaluate model performances
	history = model.evaluate(valid_images, valid_true, verbose=0)

	return history, nb_images

def resize_images(images, width, height):
	imgs = []
	for image in images:
		if np.shape(image) != (height, width, 3):
			img = cv2.resize(image, (width, height))
		else:
			img = image
		imgs.append(img)
	return imgs



def process_video_c(model, video_path, width, height, output_dir):

	model_class = tf.keras.models.load_model("E:/Documents/Cours/SeaTech/2A/Git/App/RCNN_class.keras")
	model_reg = tf.keras.models.load_model("E:/Documents/Cours/SeaTech/2A/Git/App/RCNN_reg.keras")

	cap = cv2.VideoCapture(video_path)

	cpt = 0
	while cap.isOpened():
		_, frame = cap.read()

		if len(np.shape(frame)) != 3:
			break

		img_height, img_width, _ = np.shape(frame)
		img_to_process = resize_images([frame], width, height)

		label, box = process_images(model_class, model_reg, np.array(img_to_process), width, height)

		if label:
			cpt += 1
			color = (255, 0, 0)
			box = box[0]
			rect_frame = frame.copy()
			cv2.rectangle(rect_frame, (int(box[0]/width*img_width), int(box[1]/height*img_height)), (int(box[2]/width*img_width), int(box[3]/height*img_height)), color, 2)
			cv2.imwrite(output_dir + "\\" + str(cpt) + ".jpg", frame)
			cv2.imwrite(output_dir + "\\" + str(cpt) + "_with_box.jpg", rect_frame)

	cap.release()



def process_images_c(model_class, model_reg, images, width, height):
	"""
	## Description
		Process images with the neural network created
		
	### Args:
		model (tf_model): neural network
		images (np_array): Array of all images to be processed
		width (float): width of the images
		height (float): height of the images
	
	### Returns:
		(list, np_array): labels and boxes of the processed images
	"""
	normalized_images = images / 255.0

	labels = model_class(normalized_images, training = False)

	boxes = np.zeros((len(labels), 4))

	normalized_boxes = model_reg(normalized_images, training = False)
	## Get back the right width and height
	boxes[:,0], boxes[:,2] = normalized_boxes[:,0]*width, normalized_boxes[:,2]*width
	boxes[:,1], boxes[:,3] = normalized_boxes[:,1]*height, normalized_boxes[:,3]*height

	return [np.argmax(l) for l in labels], boxes


def process_images(model, images):
	normalized_images = images / 255.0

	labels = model(normalized_images, training = False)
	print(labels)

	return [np.argmax(l) for l in labels]


def process_video(model, video_path, width, height, output_dir):
	cap = cv2.VideoCapture(video_path)

	cpt = 0
	while cap.isOpened():
		_, frame = cap.read()

		if len(np.shape(frame)) != 3:
			break

		img_to_process = resize_images([frame], width, height)

		label = process_images(model, np.array(img_to_process))

		if label[0] == 1:
			cpt += 1
			cv2.imwrite(output_dir + "\\" + str(cpt) + ".jpg", frame)
		else:
			cpt += 1
			frame = cv2.circle(frame, (20, 20), 10, color=(0,0,255) ,thickness=-1)
			cv2.imwrite(output_dir + "\\" + str(cpt) + "nofish.jpg", frame)

	cap.release()