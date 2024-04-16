## Import modules
import os
import cv2
import tensorflow as tf
from keras import layers, Model

import matplotlib.pyplot as plt
import numpy as np


### Reduire en noir et blanc
### Rescale de la saturation


fabien_path = "E:/Documents/Cours/SeaTech/2A/Projet/OFD_v4"
yohan_path = "C:/seatech/moca_2A/_PROJET/OFD_v3_bw"

## Define constants
PATH = yohan_path
SAVING_PATH = os.getcwd() + '/Models/'
CLASS_NB = 2 		# Number of classes to be recognize
EPOCHS = 50			# Number of epoch for trainning
BATCH_SIZE = 64		# Number of images in a batch
HEIGHT = 144		# width of images in px
WIDTH = 256			# height of images in px
FUNCTION = "max"
NNEURONS = 64
NLAYER = 2
MODEL_NAME = "RCNN_DS_BS" + str(BATCH_SIZE) + '_F' + FUNCTION + '_E' + str(EPOCHS) + '_N' + str(NNEURONS)  + '_NL' + str(NLAYER) + '_BW'

## CREATING THE NEURAL NETWORK MODEL
def build_feature_extractor(inputs, width, height):
	"""
	## Description
		Build the CNN part of the neural network
		
	### Args:
		inputs (tf.layers.Inputs): Give the size of the inputs
		width (float): width of the images
		height (float): height of the images
	
	### Returns:
		tf_model: CNN part of the neural network
	"""

	x = layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(int(width), int(height), 1))(inputs)
	x = layers.MaxPooling2D(2,2)(x)

	x = layers.Conv2D(64, kernel_size=3, activation = 'relu')(x)
	x = layers.MaxPooling2D(2,2)(x)

	x = layers.Conv2D(64, kernel_size=3, activation = 'relu')(x)
	x = layers.MaxPooling2D(2,2)(x)

	return x

def build_model_adaptor(inputs):
	"""
	## Description
		Build the adaptator of the model to distrubute in both heads		
	
	### Args:
		inputs (tf_model): a part of the model
	
	### Returns:
		tf_model:
	"""

	x = layers.Flatten()(inputs)
	for _ in range(NLAYER):
		x = layers.Dense(NNEURONS, activation='relu')(x)
	return x

def build_classifier_head(inputs):
	"""
	## Description
		Build the classifier head of the model

		The classifier head gives the type of object in the image
	"""
	return layers.Dense(CLASS_NB, activation='softmax', name = 'classifier_head')(inputs)

def build_model(inputs, width, height):
	"""
	## Description
		Build the entire model with its differents parts
	
	### Args:
		inputs (tf.layers.Inputs): Give the size of the inputs
		width (float): width of the images
		height (float): height of the images
	
	### Returns:
		tf_model: The neural network
	"""

	feature_extractor = build_feature_extractor(inputs, width, height)
	
	model_adaptor = build_model_adaptor(feature_extractor)
	
	classifier_output = build_classifier_head(model_adaptor)
	

	model = Model(inputs= inputs, outputs=classifier_output)

	## Choose optimizer
	model.compile(optimizer=tf.keras.optimizers.Adam(), 
		loss = {'classifier_head' : 'sparse_categorical_crossentropy'}, 
		metrics = {'classifier_head' : 'accuracy'})
	
	return model


## LOADING THE TRAINING DATA
def load_training_data(path):
	train_images = []
	train_labels = []

	with open(path + "/train/data.csv", 'r') as data_file:
		data_file.readline()        # Pass header line
		for line in data_file:
			line = line.strip().split(',')
			image_path = line[0]
			train_images.append(cv2.cvtColor(cv2.imread(path + "/train/" + image_path), cv2.COLOR_BGR2GRAY))
			train_labels.append([int(line[3])])

	return np.array(train_images), np.array(train_labels)

def load_testing_data(path):
	test_images = []
	test_labels = []

	with open(path + "/test/data.csv", 'r') as data_file:
		data_file.readline()        # Pass header line
		for line in data_file:
			line = line.strip().split(',')
			image_path = line[0]
			test_images.append(cv2.cvtColor(cv2.imread(path + "/test/" + image_path), cv2.COLOR_BGR2GRAY))
			test_labels.append([int(line[3])])
	
	return np.array(test_images), np.array(test_labels)

def load_validation_data(path):
	valid_images = []
	valid_labels = []

	with open(path + "/valid/data.csv", 'r') as data_file:
		data_file.readline()        # Pass header line
		for line in data_file:
			line = line.strip().split(',')
			image_path = line[0]
			valid_images.append(cv2.imread(path + "/valid/" + image_path))
			valid_labels.append([int(line[3])])

	return np.array(valid_images), np.array(valid_labels)

def show_images(train_images, train_labels, seed):
	"""
	## Description
		Show 16 random images in a matplotlib window with their label and boxes if there are
	
	### Args:
		train_images (list): List of images
		train_labels (_type_): List of labels associated with images
		train_boxes (_type_): List of boxes associated with images
		seed (_type_): seed of the randomness
	"""
	N = 16
	if N > train_images.shape[0]:
		N = train_images.shape[0]
	plt.figure(figsize=(20, 10))

	np.random.seed(seed)
	for cpt in range(N):
		img_id = np.random.randint(0, len(train_images)-1)
		image, label = train_images[img_id], train_labels[img_id]

		image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		text = "no_fish"
		if label == 1:
			text = "fish"

		plt.subplot(4, 4, cpt + 1)
		plt.imshow(image_color)
		plt.title(text, fontsize=8)
		plt.axis("off")


## MAKE NEURAL NETWORK READY 
def process_dataset(images, labels):
	"""
	## Description
		Prepare the dataset to be supplied to the neural network		
	
	### Args:
		images (np_array): Array of all images to be process
		labels (np_array): Array of all images labels
		boxes (np_array): Array of all images boxes
		width (float): width of the images
		height (float): height of the images
	
	### Returns:
		(np_array, np_array): images and verification for the neural networks training
	"""
	## Normalize pixel values to be between 0 and 1
	images = images / 255.0

	## Shuffling all the data
	seed = 1000
	images = tf.random.shuffle(images, seed=seed)
	labels = tf.random.shuffle(labels, seed=seed)

	return images, labels

def train_model(model, path, batch_size, epochs, checkpoints = False, checkpoints_dir = "/", checkpoints_name = "checkpoint.keras"):
	"""
	## Description
		Train the neural network
		
	### Args:
		model (tf_model): neural network
		path (str): path of the dataset
		width (float): width of the images
		height (height): height of the images
		batch_size (int): number of images in a batch
		epochs (int): number of epoch to be done to train the neural network
		checkpoints (bool, optional): Enable or disable checkpoint during trainning. Defaults to False.
		checkpoints_dir (str, optional): Checkpoint file directory. Defaults to "/".
		checkpoints_name (str, optional): Checkpoint file name. Defaults to "checkpoint.keras".
	
	### Returns:
		(tf_model, dict): the model trained and the history of its trainning
	"""
	## Loading training data
	train_images, train_labels = load_training_data(path)
	test_images, test_labels = load_testing_data(path)

	## Processing training data
	train_images, train_labels = process_dataset(train_images, train_labels)
	test_images, test_labels = process_dataset(test_images, test_labels)

	## Training and testing set informations
	print("Number of image in the training set :", len(train_images))
	print("Number of image in the testing set :", len(test_images))

	if checkpoints:
		# Create a callback that saves the model's weights
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_dir+checkpoints_name,
														save_weights_only=False,
														verbose=1)

		## Train the model with checkpoints
		history = model.fit(train_images, train_labels, 
							validation_data=(test_images, test_labels), 
							batch_size=batch_size, epochs=epochs, 
							callbacks=[cp_callback])
	else:
		## Train the model without checkpoints
		history = model.fit(train_images, train_labels, 
							validation_data=(test_images, test_labels), 
							batch_size=batch_size, epochs=epochs, verbose=1)
	
	## Return the model and the history of its training
	return model, history

def evaluate_model(model, path):
	"""
	## Description
		Evaluate the neural network with validation data
	
	### Args:
		model (tf_model): neural network
		path (str): path to the validation data
		width (float): width of the images
		height (height): height of the images
	
	### Returns:
		list: history of the evaluation process
	"""
	## Loading validation data
	valid_images, valid_labels = load_validation_data(path)

	## Validation set informations
	print("Number of image in the validation set :", len(valid_images))

	## Processing validation data
	valid_images, valid_labels = process_dataset(valid_images, valid_labels)

	## Evaluate model performances
	history = model.evaluate(valid_images, valid_labels, verbose=0)

	return history

def save_model(model, model_name, path = "/"):
	model.save(path + model_name + ".keras")

def load_model(path, model_name):
	return tf.keras.models.load_model(path + model_name + ".keras")

def plot_history(history):
	"""
	## Description
		Plot the the history of the trainning process
	
	### Args:
		history (dict): Trainning history dictionnary of the trainning model
	"""
	with open(SAVING_PATH + "history.txt", 'a') as file:
		file.write(MODEL_NAME + ',' + str(EPOCHS) + ',' + str(BATCH_SIZE) + ',' + str(history.history.keys()) + '\n')
		for value in history.history["accuracy"]:
			file.write(f"{value:8.5f},")
		file.write("\n")
		for value in history.history["loss"]:
			file.write(f"{value:8.5f},")
		file.write("\n")
		for value in history.history["val_accuracy"]:
			file.write(f"{value:8.5f},")
		file.write("\n")
		for value in history.history["val_loss"]:
			file.write(f"{value:8.5f},")
		file.write("\n")	

	
	plt.plot(history.history['accuracy'], label='classifier accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')

	plt.show()
	
def process_images(model, images):
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

	labels = model(normalized_images, training = False)

	return [np.argmax(l) for l in labels]


## Build the neural network model
model = build_model(layers.Input(shape=(HEIGHT, WIDTH,1, )), WIDTH, HEIGHT)

## Show a summary of the neural network
model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

model, history = train_model(model, PATH, BATCH_SIZE, EPOCHS, checkpoints=False, checkpoints_dir=SAVING_PATH)
save_model(model, MODEL_NAME, SAVING_PATH)
plot_history(history)

## Load old model
# model = load_model(SAVING_PATH, "RCNN_sofmax")
# loss, acc, mse = evaluate_model(model, PATH, WIDTH, HEIGHT)
# print(f"Restored model, loss: {loss:5.2f}; accuracy: {acc:5.2f}; mse: {mse:5.2f}")


## Process images
# valid_images, valid_labels, valid_boxes = load_validation_data(PATH)
# labels, boxes = process_images(model, valid_images, WIDTH, HEIGHT)

## Show random images with boxes and label
# show_images(valid_images, valid_labels, valid_boxes, seed=10)
# show_images(valid_images, labels, boxes, seed=10)

## Show plots
# plt.show()