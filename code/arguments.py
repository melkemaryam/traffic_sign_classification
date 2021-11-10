import argparse


class Args:
	
	# create argument parser
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", required=True, help="path to output model")
	ap.add_argument("-d", "--dataset", required=True, help="path to input GTSRB")
	ap.add_argument("-i", "--images", required=True, help="path to testing directory containing images")
	ap.add_argument("-pr", "--predictions", required=True, help="path to output predictions directory")
	ap.add_argument("-pl", "--plot", type=str, default="plot.png", help="path to training history plot")
	self.args = vars(ap.parse_args())