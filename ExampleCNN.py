from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.callbacks import CSVLogger

import numpy as np

import keras

# Example of a CNN trained on the Feature Input on Dataset 1

class CNN(object):
	def __init__(self):
		self.width = 8
		self.height = 8
		self.channels = 16
		self.classes = 3
		self.epochs = 100
		self.batch_size = 128
		self.opt = SGD(lr=0.01)
		self.activation = "elu"
		self.lossFunction = "categorical_crossentropy"

	def LoadPositions(self):
		return np.load('CnnPositions.npy')

	def LoadLabels(self):
		return np.load('CnnLabels.npy')

	def OneHotEncoding(self, y):
		
		encoder = LabelEncoder()
		encoder.fit(y)
		encoded_y = encoder.transform(y)

		final_y = np_utils.to_categorical(encoded_y, self.classes)

		return final_y

	def ShapeData(self, Positions):

		ShapedChessPositions = np.reshape(self.Positions, (self.Positions.shape[0], self.width, self.height, self.channels))

		return ShapedChessPositions

	def SplitDataset(self):
		trainData, testData, trainLabels, testLabels = train_test_split(self.ConvPositions, self.Labels, test_size=0.1, random_state=42)

		return [trainData, testData, trainLabels, testLabels]

	def NeuralNet(self):
		
		self.model = Sequential()
		self.model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(self.width, self.height, self.channels)))
		self.model.add(Activation(self.activation))
		self.model.add(Convolution2D(50, 3, 3, border_mode="same")) 
		self.model.add(Activation(self.activation))
		self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(250, activation=self.activation))
		self.model.add(Dense(self.classes))
		self.model.add(Activation('softmax'))

		self.model.compile(loss=self.lossFunction, optimizer=self.opt, metrics=['accuracy'])

		self.model.fit(self.TrainPositions, self.TrainLabels, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1, validation_data=(self.TestingPositions, self.TestLabels))
		
		self.model.fit(self.TrainPositions, self.TrainLabels)

		self.score = self.model.evaluate(self.TestingPositions, self.TestLabels, verbose=0)
		print('Test accuracy:', self.score[1])

		self.model.save_weights('CNNWeights.h5')

	def main(self):
		self.Positions = self.LoadPositions()
		self.Evaluations = self.LoadLabels() 
		self.ConvPositions = self.ShapeData(self.Positions)
		self.Labels = self.OneHotEncoding(self.Evaluations)
		self.FinalDataset = self.SplitDataset()

		self.TrainPositions = self.FinalDataset[0]
		self.TestingPositions = self.FinalDataset[1]
		self.TrainLabels = self.FinalDataset[2]
		self.TestLabels = self.FinalDataset[3]

		print "Start of Experiment!"

		self.NeuralNet()

if __name__ == '__main__':

	ConvNet = CNN()
	ConvNet.main()	
