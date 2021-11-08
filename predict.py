# USAGE
# python predict.py --model output/neural_net.model --images gtsrb-german-traffic-sign/Test --predictions predictions

# import packages
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to pre-trained traffic sign recognizer")
ap.add_argument("-i", "--images", required=True, help="path to testing directory containing images")
ap.add_argument("-p", "--predictions", required=True, help="path to output predictions directory")
args = vars(ap.parse_args())

# load the traffic sign recognizer model
print("[INFO] loading model...")
model = load_model(args["model"])

# load the label names
sign_names = open("sign_names.csv").read().strip().split("\n")[1:]
sign_names = [s.split(";")[1] for s in sign_names]

# grab the paths to the input images, shuffle them, and grab a sample
print("[INFO] predicting...")
paths_to_image = list(paths.list_images(args["images"]))
random.shuffle(paths_to_image)
paths_to_image = paths_to_image[:25]

# loop over the image paths
for (i, path_to_image) in enumerate(paths_to_image):
	
	# load the image, resize it to 32x32 pixels, and then apply
	# Contrast Limited Adaptive Histogram Equalization (CLAHE),
	# just like we did during training
	image = io.imread(path_to_image)
	image = transform.resize(image, (32, 32))
	image = exposure.equalize_adapthist(image, clip_limit=0.1)

	# preprocess the image by scaling it to the range [0, 1]
	image = image.astype("float32") / 255.0
	image = np.expand_dims(image, axis=0)

	# make predictions using the traffic sign recognizer CNN
	predictions = model.predict(image)
	j = predictions.argmax(axis=1)[0]
	label = sign_names[j]

	# load the image using OpenCV, resize it, and draw the label
	image = cv2.imread(path_to_image)
	image = imutils.resize(image, width=128)
	cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# save the image to disk
	p = os.path.sep.join([args["predictions"], "{}.png".format(i)])
	cv2.imwrite(p, image)