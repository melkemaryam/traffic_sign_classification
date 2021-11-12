# Usage

# python main.py --model ../output/neural_net.model --dataset ../gtsrb-german-traffic-sign --images ../gtsrb-german-traffic-sign/Test --predictions ../predictions --plot ../output/plot.png


from arguments import Args
from train import Train_Net
from predict import Predict_Net

if __name__ == '__main__':
	try:
		
		a = Args()
		a.parse_arguments()

		# create objects of training and predicting classes
		tr = Train_Net()
		p = Predict_Net()

		# start training
		tr.main_train_net()

		# continue with predicting once training is done
		p.main_predict_net()

	except KeyboardInterrupt:
		pass