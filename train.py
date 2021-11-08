# USAGE
# python train.py --dataset gtsrb-german-traffic-sign --model output/neural_net.model --plot output/plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# import packages
from neural_net import Neural_Net

import tensorflow.keras

import sklearn
import skimage

import numpy as np
import argparse

import random
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input GTSRB")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to training history plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64

def load_images(pwd, path_to_csv):

	labels = []
	data = []

	rows = open(path_to_csv).read().strip().split("\n")[1:]
	random.shuffle(rows)

	# loop over the rows of the CSV file
	for (i, row) in enumerate(rows):
		
		# check for status update
		if i > 0 and i % 1000 == 0:
			print("[INFO] processed {} total images".format(i))

		# split the row into components
		# then grab the class ID and image path
		(label, path_to_image) = row.strip().split(";")[-2:]

		# derive the full path to the image file and load it
		path_to_image = os.path.sep.join([pwd, path_to_image])
		image = io.imread(path_to_image)

		# resize the image to be 32x32 pixels, ignoring aspect ratio,
		# and then perform Contrast Limited Adaptive Histogram
		# Equalization (CLAHE)
		image = transform.resize(image, (32, 32))
		image = exposure.equalize_adapthist(image, clip_limit=0.1)

		# update the list of data and labels, respectively
		data.append(image)
		labels.append(int(label))

	# convert the data and labels to NumPy arrays
	data = np.array(data)
	labels = np.array(labels)

	# return a tuple of the data and labels
	return (data, labels)

# load the sign names
sign_names = open("sign_names.csv").read().strip().split("\n")[1:]
sign_names = [s.split(";")[1] for s in sign_names]

# derive the path to the training and testing CSV files
path_to_train = os.path.sep.join([args["dataset"], "Train.csv"])
path_to_test = os.path.sep.join([args["dataset"], "Test.csv"])

# load the training and testing data
print("[INFO] loading training and testing data...")
(train_X, train_Y) = load_split(args["dataset"], path_to_train)
(test_X, test_Y) = load_split(args["dataset"], path_to_test)

# scale data to the range of [0, 1]
train_X = train_X.astype("float32") / 255.0
test_X = test_X.astype("float32") / 255.0

# one-hot encode the training and testing labels
num_signs = len(np.unique(train_Y))
train_Y = to_categorical(train_Y, num_signs)
test_Y = to_categorical(test_Y, num_signs)

# calculate the total number of images in each class and
# initialize a dictionary to store the class weights
total_images_class = train_Y.sum(axis=0)
total_weight_class = dict()

# loop over all classes and calculate the class weight
for i in range(0, len(total_images_class)):
	total_weight_class[i] = total_images_class.max() / total_images_class[i]

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")


# initialize the optimizer and compile the model
print("[INFO] compiling model...")
optimiser = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = Neural_Net.create_net(w=32, h=32, d=3, signs=num_signs)
model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
train = model.fit(
	aug.flow(train_X, train_Y, batch_size=BS),
	validation_data=(test_X, test_Y),
	steps_per_epoch=train_X.shape[0] // BS,
	epochs=NUM_EPOCHS,
	class_weight=total_weight_class,
	verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=BS)
print(classification_report(test_Y.argmax(axis=1),
	predictions.argmax(axis=1), target_names=sign_names))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# plot the training loss and accuracy
num = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()

plt.plot(num, train.history["loss"], label="train_loss")
plt.plot(num, train.history["val_loss"], label="val_loss")
plt.plot(num, train.history["accuracy"], label="train_acc")
plt.plot(num, train.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
