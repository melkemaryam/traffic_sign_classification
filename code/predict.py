# USAGE
# python predict.py --model output/neural_net.model --images gtsrb-german-traffic-sign/Test --predictions predictions

# import packages
import argparse
from arguments import Args
import cv2
import imutils
from imutils import paths
import numpy as np
import os
import random
from skimage import transform
from skimage import exposure
from skimage import io
from tensorflow.keras.models import load_model
from train import Train_Net


class Predict_Net:

	tr = Train_Net()
	a = Args()

	def load_net(self):

		# load the trained model
		print("[INFO] loading model...")
		self.model = load_model(a.args["model"])


	def prediction_process(self):

		# grab the paths to the input images, shuffle them, and grab a sample
		print("[INFO] predicting...")

		sign_names = tr.get_sign_names()

		self.paths_to_image = list(paths.list_images(a.args["images"]))
		random.shuffle(self.paths_to_image)

		# choose only 24 images
		self.paths_to_image = self.paths_to_image[:25]

		# loop over the image paths
		for (i, path_to_image) in enumerate(paths_to_image):
			
			self.save_image(
				self.write_on_image(
					self.predict(
						self.process_image(
							path_to_image),
						sign_names),
					path_to_image))



	def process_image(self, path_to_image):

		# resize images and perform CLAHE
		image = io.imread(path_to_image)
		image = transform.resize(image, (32, 32))
		image = exposure.equalize_adapthist(image, clip_limit=0.1)

		# preprocess the image by scaling it to the range [0, 1]
		image = image.astype("float32") / 255.0
		image = np.expand_dims(image, axis=0)

		return image

	def predict(self, sign_names, image):

		# make predictions using the traffic sign recognizer CNN
		predictions = self.model.predict(image)
		j = predictions.argmax(axis=1)[0]
		label = sign_names[j]

		return image

	def write_on_image(self, image, path_to_image):

		# load the image using OpenCV, resize it, and draw the label
		image = cv2.imread(path_to_image)
		image = imutils.resize(image, width=128)
		cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		return image

	def save_image(self, image):

		# save the image to disk
		p = os.path.sep.join([a.args["predictions"], "{}.png".format(i)])
		cv2.imwrite(p, image)