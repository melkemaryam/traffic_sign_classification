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

	
	def main_predict_net(self):

		self.tr = Train_Net()
		arg = Args()
		self.args = arg.parse_arguments()

		self.load_net()
		self.prediction_process()


	def load_net(self):

		# load the trained model
		print("[INFO] loading model...")
		self.model = load_model(self.args["model"])


	def prediction_process(self):

		# grab the paths to the input images, shuffle them, and grab a sample
		print("[INFO] predicting...")

		sign_names = self.tr.get_sign_names()

		paths_to_image = list(paths.list_images(self.args["images"]))
		random.shuffle(paths_to_image)

		# choose only 30 images
		paths_to_image = paths_to_image[:31]

		# loop over the image paths
		for (i, path_to_image) in enumerate(paths_to_image):
			
			# resize images and perform CLAHE
			image = io.imread(path_to_image)
			image = transform.resize(image, (32, 32))
			image = exposure.equalize_adapthist(image, clip_limit=0.1)

			# preprocess the image by scaling it to the range [0, 1]
			image = image.astype("float32") / 255.0
			image = np.expand_dims(image, axis=0)

			# make predictions using the traffic sign recognizer CNN
			predictions = self.model.predict(image)
			j = predictions.argmax(axis=1)[0]
			label = sign_names[j]

			# load the image using OpenCV, resize it, and draw the label
			image = cv2.imread(path_to_image)
			image = imutils.resize(image, width=128)
			cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			# save the image to disk
			#CHANGE: _rl = turn right/left only, _all = all signs
			p = os.path.sep.join([self.args["predictions_all"], "{}.png".format(i)])
			cv2.imwrite(p, image)
