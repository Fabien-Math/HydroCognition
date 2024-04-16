import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from util import convex_hull
import cv2

## Define constants
PATH = "Your/Path"
SAVING_PATH = os.getcwd() + '/Models/'
MODEL_NAME = "RCNN_DS_BS64_Fmax_E50_N64_NL2"
HEIGHT = 144		# width of images in px
WIDTH = 256			# height of images in px
SUBDIVISION = 4

# Your image path
image_path = os.getcwd() + "/white.jpg"
image = cv2.imread(image_path)

IMG_HEIGHT, IMG_WIDTH, _ = np.shape(image)

H_STRIDE = IMG_HEIGHT // SUBDIVISION
W_STRIDE = IMG_WIDTH // SUBDIVISION
# image = cv2.resize(image, (WIDTH, HEIGHT))

def load_model(path, model_name):
	return tf.keras.models.load_model(path + model_name + ".keras")

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

	return labels[:,1]


# Subdivide the image
sub_images = []
for k in range(1, SUBDIVISION):
    for j in range(SUBDIVISION + 1 - k):
        for i in range(SUBDIVISION + 1 - k):
            sub_images.append(image[j*H_STRIDE:(j+k)*H_STRIDE, i*W_STRIDE:(i+k)*W_STRIDE])
sub_images.append(image)

# Resize all subimage to be feed in the CNN
resized_sub_images = []
for sub_image in sub_images:
    resized_sub_images.append(cv2.resize(sub_image, (WIDTH, HEIGHT)))
resized_sub_images = np.array(resized_sub_images)

# Load pre-trainned CNN model
model = load_model(SAVING_PATH, MODEL_NAME)

# List of all prediction for all sub images
preds = np.array(process_images(model, resized_sub_images))

# Threshold for occurence of a fish
thresh_fnf = 0.5
if preds[-1] < thresh_fnf:
    print("\nNO FISH IN THE IMAGE !")
    
    # End the program because no fish is found in the image
    # exit()
    

# Initialize matrix
interest_matrix = np.zeros((SUBDIVISION, SUBDIVISION))
normalization_matrix = np.zeros((SUBDIVISION, SUBDIVISION))
step_nm = np.zeros((SUBDIVISION, SUBDIVISION))
old_im = np.zeros((SUBDIVISION, SUBDIVISION))

# Build the interest and the normalisation matrix
preds_id = 0
for k in range(SUBDIVISION, 0, -1):
    for j in range(k):
        for i in range(k):
            stride = SUBDIVISION - k + 1 # Square root of the number of 16th of the image in the sub image
            interest_matrix[j : j + stride, i : i + stride] += preds[preds_id]
            normalization_matrix[j : j + stride, i : i + stride] += 1
            preds_id += 1

    # Normalization of the probability
    new_im = interest_matrix-old_im
    step_nm = normalization_matrix - step_nm
    new_im /= step_nm
    new_im /= np.max(new_im)
    step_nm = np.copy(normalization_matrix)
    interest_matrix = new_im + old_im
    old_im = np.copy(interest_matrix)
interest_matrix /= SUBDIVISION

# Multiplication of 16th of the image probability to give importance to their probability
interest_matrix *= np.reshape(preds[0:16], (SUBDIVISION, SUBDIVISION))
# Display interest_matrix
print(interest_matrix)

# Parameter for the threshold of fish in the image
theta = 0.5

# Relativ threshold to locate the fish in the image
thresh = np.min(interest_matrix) * (1 - theta) + np.max(interest_matrix) * theta
interest_matrix = np.where(interest_matrix >= thresh, interest_matrix, 0)

# Display all sub images in a plot 
plt.figure("Sub images")
plt.suptitle("Sub-images for the CNN", fontsize=25)
for i, sub_image in enumerate(resized_sub_images):
    plt.subplot(5, 6, i+1)
    plt.axis("off")
    plt.title(f"Proba : {preds[i]:1.3f}")
    plt.imshow(cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB))

# Display the result with red square where fish are
plt.figure("Results")
points = []
for j, rows in enumerate(interest_matrix):
    for i, value in enumerate(rows):
        if value >= thresh:
            points.append((i*IMG_WIDTH/SUBDIVISION, j*IMG_HEIGHT/SUBDIVISION))
            points.append(((i+1)*IMG_WIDTH/SUBDIVISION, j*IMG_HEIGHT/SUBDIVISION))
            points.append(((i+1)*IMG_WIDTH/SUBDIVISION, (j+1)*IMG_HEIGHT/SUBDIVISION))
            points.append((i*IMG_WIDTH/SUBDIVISION, (j+1)*IMG_HEIGHT/SUBDIVISION))
hull = np.array(convex_hull(points))
plt.axis("off")
plt.plot(hull[:,0], hull[:,1], color='red', linewidth=10)
# plt.plot(np.array([i, i+1, i+1, i, i])*IMG_WIDTH/SUBDIVISION, np.array([j, j, j+1, j+1, j])*IMG_HEIGHT/SUBDIVISION, color='red')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
# print(preds)
plt.show()