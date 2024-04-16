## Import modules
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import tensorflow as tf
from keras import datasets, layers, Model

import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


fabien_path = "E:/Documents/Cours/SeaTech/2A/Projet/OFD_v3"
yohan_path = "C:/seatech/moca_2A/_PROJET/OFD_v3"

## Define constants
# PATH = yohan_paths
SAVING_PATH = os.getcwd() + '/Models/'
CLASS_NB = 10 		# Number of classes to be recognize
EPOCHS = 10			# Number of epoch for trainning
BATCH_SIZE = 32		# Number of images in a batch
HEIGHT = 32		# width of images in px
WIDTH = 32			# height of images in px
FUNCTION = "max"
NNEURONS = 64
NLAYER = 2
# MODEL_NAME = "RCNN_DS_" + str(BATCH_SIZE) + '_' + FUNCTION + '_' + str(NNEURONS)
MODEL_NAME = "RCNN_CIFAR_BS" + str(BATCH_SIZE) + '_F' + FUNCTION + '_E' + str(EPOCHS) + '_N' + str(NNEURONS)  + '_NL' + str(NLAYER)


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

	# x = layers.MaxPooling2D(2,2)(inputs)
	# x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)

	x = layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(int(width), int(height), 3))(inputs)
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


## MAKE NEURAL NETWORK READY 
def train_model(model, batch_size, epochs, checkpoints = False, checkpoints_dir = "/", checkpoints_name = "checkpoint.keras"):
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

def evaluate_model(model):
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
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

	return test_loss, test_acc

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
model = build_model(layers.Input(shape=(HEIGHT, WIDTH, 3,)), WIDTH, HEIGHT)

## Show a summary of the neural network
model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

model, history = train_model(model, BATCH_SIZE, EPOCHS, checkpoints=False, checkpoints_dir=SAVING_PATH)
save_model(model, MODEL_NAME, SAVING_PATH)
plot_history(history)

## Load old model
# model = load_model(SAVING_PATH, MODEL_NAME)
# l = evaluate_model(model, PATH)
# print(l)
# print(f"Restored model, loss: {loss:5.2f}; accuracy: {acc:5.2f}; mse: {mse:5.2f}")


## Process images
# valid_images, valid_labels, valid_boxes = load_validation_data(PATH)
# labels, boxes = process_images(model, valid_images, WIDTH, HEIGHT)

## Show random images with boxes and label
# show_images(valid_images, valid_labels, valid_boxes, seed=10)
# show_images(valid_images, labels, boxes, seed=10)

## Show plots
# plt.show()