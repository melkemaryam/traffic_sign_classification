# USAGE
# python train.py --dataset gtsrb-german-traffic-sign --model output/neural_net.model --plot output/plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# import packages
import argparse
from arguments import Args
import numpy as np
from neural_net import Neural_Net
import os
import random
from skimage import exposure
from skimage import io
from skimage import transform
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


a = Args()

class Train_Net:

	
	def main_train_net(self):

		# initialise parameters
		number_of_epochs = 30
		initial_learning_rate = 1e-3
		batch_size = 64

		# prepare the data and the model
		self.prepare_data()

		# train the model
		self.train(batch_size, number_of_epochs)

		# evaluate the model
		self.evaluate(batch_size)

		# save data in a plot
		self.save_data(number_of_epochs)


	def load_images(self, pwd, path_to_csv):

	labels = []
	data = []

	# load data
	rows = open(path_to_csv).read().strip().split("\n")[1:]
	random.shuffle(rows)

	# loop over the rows of the CSV file
	for (i, row) in enumerate(rows):
		
		# print status update
		if i > 0 and i % 100 == 0:
			print("[INFO] processed {} total images".format(i))

		# get classId and path to image
		(label, path_to_image) = row.strip().split(";")[-2:]

		# create full path to image
		path_to_image = os.path.sep.join([pwd, path_to_image])
		image = io.imread(path_to_image)

		# resize the image and perform CLAHE
		image = transform.resize(image, (32, 32))
		image = exposure.equalize_adapthist(image, clip_limit=0.1)

		# update the list of data and labels
		data.append(image)
		labels.append(int(label))

	# convert the data and labels to NumPy arrays
	data = np.array(data)
	labels = np.array(labels)

	# return a tuple of the data and labels
	return (data, labels)


	def get_sign_names(self):
		
		# load sign names
		sign_names = open("sign_names.csv").read().strip().split("\n")[1:]
		sign_names = [s.split(";")[1] for s in sign_names]

		return sign_names

	def prepare_data(self):

		# derive the path to the training and testing CSV files
		path_to_train = os.path.sep.join([a.args["dataset"], "Train.csv"])
		path_to_test = os.path.sep.join([a.args["dataset"], "Test.csv"])

		# load the training and testing data
		print("[INFO] loading training and testing data...")
		(self.train_X, self.train_Y) = load_images(a.args["dataset"], path_to_train)
		(self.test_X, self.test_Y) = load_images(a.args["dataset"], path_to_test)

		# scale data to the range of [0, 1]
		self.train_X = self.train_X.astype("float32") / 255.0
		self.test_X = self.test_X.astype("float32") / 255.0

		# one-hot encode the training and testing labels
		self.num_signs = len(np.unique(self.train_Y))
		self.train_Y = to_categorical(self.train_Y, self.num_signs)
		self.test_Y = to_categorical(self.test_Y, self.num_signs)

		# calculate the total number of images in each class and
		# initialize a dictionary to store the class weights
		self.total_images_class = self.train_Y.sum(axis=0)
		self.total_weight_class = dict()

		# loop over all classes and calculate the class weight
		for i in range(0, len(self.total_images_class)):
			self.total_weight_class[i] = self.total_images_class.max() / self.total_images_class[i]

		self.compile()

	def get_augmentation(self):
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

		return aug

	def optimise(self):

		# initialise the optimiser
		optimiser = Adam(
			learning_rate=initial_learning_rate, 
			decay=initial_learning_rate / (number_of_epochs * 0.5))

		return optimiser

	def compile(self):
		# compile the model
		print("[INFO] compiling model...")
		opt = self.optimise()

		self.model = Neural_Net.create_net(w=32, h=32, d=3, signs=self.num_signs)
		self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


	def train(self, batch_size, number_of_epochs):
		# train the network
		print("[INFO] training network...")

		augment = self.get_augmentation()
		train = self.model.fit(
			augment.flow(self.train_X, self.train_Y, batch_size=batch_size),
			validation_data=(self.test_X, self.test_Y),
			steps_per_epoch=self.train_X.shape[0] // batch_size,
			epochs=number_of_epochs,
			class_weight=self.total_weight_class,
			verbose=1)

	def evaluate(self, batch_size):

		# evaluate the network
		print("[INFO] evaluating network...")

		sign_names = self.get_sign_names()
		predictions = self.model.predict(self.test_X, batch_size=batch_size)
		print(classification_report(self.test_Y.argmax(axis=1), predictions.argmax(axis=1), target_names=sign_names))

	def save_data(self, number_of_epochs):
		
		# save the network to disk
		print("[INFO] serializing network to '{}'...".format(a.args["model"]))
		self.model.save(a.args["model"])

		# plot the training loss and accuracy
		num = np.arange(0, number_of_epochs)
		plt.style.use("ggplot")
		plt.figure()

		# write plot
		plt.plot(num, train.history["loss"], label="train_loss")
		plt.plot(num, train.history["val_loss"], label="val_loss")
		plt.plot(num, train.history["accuracy"], label="train_acc")
		plt.plot(num, train.history["val_accuracy"], label="val_acc")

		# establish legend
		plt.title("Training Loss and Accuracy on Dataset")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig(a.args["plot"])
