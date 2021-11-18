# import tensorflow packages
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential


class Neural_Net:

	def create_net(self, w, h, d, signs):

		#initialise model
		self.model = Sequential()
		input_shape = (h, w, d)

		# first convolution
		self.add_first_convolution(8, 5, input_shape)

		# pooling
		self.add_pooling()

		# second convolution
		self.add_convolution(16, 3)

		# third convolution
		self.add_convolution(16, 3)

		# pooling
		self.add_pooling()

		# fourth convolution
		self.add_convolution(32, 3)

		# fifth convolution
		self.add_convolution(32, 3)

		# pooling
		self.add_pooling()

		self.model.add(Flatten())
		# first fully connected layer
		self.add_fully_connected()

		# second fully connected layer
		self.add_fully_connected()

		# softmax classifier
		self.add_softmax(signs)
		
		# return the constructed network architecture
		return self.model



	def add_first_convolution(self, filter_size, kernel_size, input_shape):

		channel_dimension = -1 #Channels Last. Image data is represented in a three-dimensional array where the last channel represents the color channels, e.g. [rows][cols][channels].

		self.model.add(Conv2D(filter_size, (kernel_size, kernel_size), padding="same", input_shape=input_shape))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=channel_dimension))

	def add_convolution(self, filter_size, kernel_size):

		channel_dimension = -1 #Channels Last. Image data is represented in a three-dimensional array where the last channel represents the color channels, e.g. [rows][cols][channels].

		self.model.add(Conv2D(filter_size, (kernel_size, kernel_size), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=channel_dimension))

	def add_pooling(self):

		self.model.add(MaxPooling2D(pool_size=(2, 2)))

	def add_fully_connected(self):

		self.model.add(Dense(128))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.5))

	def add_softmax(self, signs):

		self.model.add(Dense(signs))
		self.model.add(Activation("softmax"))
