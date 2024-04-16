import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from util import convex_hull
import cv2

def load_model(path = os.getcwd() + '/Models/', model_name = "RCNN_DS_BS64_Fmax_E50_N64_NL2"):
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
def subdivide_image(image, img_width, img_height, subdivision = 4):
    h_stride = img_height // subdivision
    w_stride = img_width // subdivision
    sub_images = []
    for k in range(1, subdivision):
        for j in range(subdivision + 1 - k):
            for i in range(subdivision + 1 - k):
                sub_images.append(image[j*h_stride:(j+k)*h_stride, i*w_stride:(i+k)*w_stride])
    sub_images.append(image)
    return sub_images

# Resize all subimage to be feed in the CNN
def resize_images(images):
    resized_sub_images = []
    for sub_image in images:
        resized_sub_images.append(cv2.resize(sub_image, (WIDTH, HEIGHT)))
    return np.array(resized_sub_images)

# Locate the fish in the image
def locate_fish(model, image, subdivision = 4):

    img_height, img_width, _ = np.shape(image)
    sub_images = subdivide_image(image, img_width, img_height)
    resized_sub_images = resize_images(sub_images)

    # List of all prediction for all sub images
    preds = np.array(process_images(model, resized_sub_images))

    # Threshold for occurence of a fish
    thresh_fnf = 0.5
    if preds[-1] < thresh_fnf:
        return "NO FISH IN THE IMAGE !"
        
        # End the program because no fish is found in the image
        # exit()
        

    # Initialize matrix
    interest_matrix = np.zeros((subdivision, subdivision))
    normalization_matrix = np.zeros((subdivision, subdivision))
    step_nm = np.zeros((subdivision, subdivision))
    old_im = np.zeros((subdivision, subdivision))

    # Build the interest and the normalisation matrix
    preds_id = 0
    for k in range(subdivision, 0, -1):
        for j in range(k):
            for i in range(k):
                stride = subdivision - k + 1 # Square root of the number of 16th of the image in the sub image
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
    interest_matrix /= subdivision

    # Multiplication of 16th of the image probability to give importance to their probability
    interest_matrix *= np.reshape(preds[0:16], (subdivision, subdivision))
    
    # Display interest_matrix
    # print(interest_matrix)

    # Parameter for the threshold of fish in the image
    theta = 0.5
    # Relative threshold to locate the fish in the image
    thresh = np.min(interest_matrix) * (1 - theta) + np.max(interest_matrix) * theta

    interest_matrix = np.where(interest_matrix >= thresh, interest_matrix, 0)

    return interest_matrix, preds

def display_subimages(sub_images, predictions):
    # Display all sub images in a plot 
    plt.figure("Sub images")
    plt.suptitle("Sub-images for the CNN", fontsize=25)
    for i, sub_image in enumerate(sub_images):
        plt.subplot(5, 6, i+1)
        plt.axis("off")
        plt.title(f"Proba : {predictions[i]:1.3f}")
        plt.imshow(cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB))

# Display the result with red square where fish are
def display_result(interest_matrix, img_width, img_height, subdivision = 4):
    # Parameter for the threshold of fish in the image
    theta = 0.5
    # Relative threshold to locate the fish in the image
    thresh = np.min(interest_matrix) * (1 - theta) + np.max(interest_matrix) * theta
    plt.figure("Results")
    points = []
    for j, rows in enumerate(interest_matrix):
        for i, value in enumerate(rows):
            if value >= thresh:
                points.append((i*img_width/subdivision, j*img_height/subdivision))
                points.append(((i+1)*img_width/subdivision, j*img_height/subdivision))
                points.append(((i+1)*img_width/subdivision, (j+1)*img_height/subdivision))
                points.append((i*img_width/subdivision, (j+1)*img_height/subdivision))
    hull = np.array(convex_hull(points))
    plt.axis("off")
    plt.plot(hull[:,0], hull[:,1], color='red', linewidth=10)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()