from NN import NeuralNetwork
import numpy as np
import pickle


def sigmoid(output):
	output = np.clip(output, -500, 500 ) # prevent overflow
	output = 1.0/(1 + np.exp(-output))
	return output

def categorical_crossentropy(predictions, targets):
    epsilon = 1e-15 # prevent log(0)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(predictions))


with open('data', 'rb') as f:
	data = pickle.load(f)

train_data = data[0]

train_X = train_data[0]
train_y = train_data[1]

input_dimension = train_X.shape[1]
output_dimension = train_y.shape[1]

network_structure = [input_dimension, 50, 50, output_dimension]

nn = NeuralNetwork(network_structure, sigmoid, sigmoid)

test_data = data[1]

nn.train(train_X, train_y, test_data)
