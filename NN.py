#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

	Classe Neural Network implémentant un réseau de neurones de type "feed forward"


	Organisation du code :
	- initialisation du réseau et des couches
	- fonction d'entraînement, propagation avant, propagation arrière
	- fonctions de test


	Notations
	- w réfère aux poids / paramètres connectant une couche à la précédente
	- z réfère aux valeurs des neurones avant activation
	- a réfère aux valeurs des neurones après activation

"""

import numpy as np
from math import log
from math import sqrt



class NeuralNetwork:

	def __init__(self, nb_neurons, activation_function, output_function):

		# -----layers init------
		self.layers = list()
		# input layer
		self.layers.append(self.Layer(0, 0, None))
		# hidden layers
		self.layers.extend([self.Layer(
			nb_neurons[x], 
			nb_neurons[x - 1], 
			activation_function
			) for x in range(1, len(nb_neurons)-1)])
		# output layer
		self.layers.append(self.Layer(nb_neurons[-1], nb_neurons[-2], output_function))


	class Layer:

		"""
			Les couches sont constituées de deux sous-couches Z_layer et A_layer (respectivement avant et après activation, conformément aux notations)
			Les vecteurs et les paramètres sont stockés dans les sous-couches, les gradients sont stockés dans la couche elle-même

			On distingue 3 types de couches différentes:
			- couche d'input, qui se définit uniquement par ses valeurs de sortie ou a_vector. Cette valeur est passée depuis l'extérieur du réseau.
			- couche cachée, connectées avec les poids (weights), avec des valeurs et une fonction d'activation pour la passe avant, et des valeurs pour la passe arrière
			- couche d'output, qui est une couche cachée avec une fonction d'activation propre

		"""
	
		def __init__(self, nb_neurons, prev_nb_neurons, activation_function):
			self.prev_nb_neurons = prev_nb_neurons
			self.nb_neurons = nb_neurons
			self.z_layer = self.Z_layer(nb_neurons, prev_nb_neurons)
			self.a_layer = self.A_layer(activation_function)
			self.update_matrix = 0
			self.dE_dz = 0

		def reset_update(self):
			self.update_matrix = 0
			self.dE_dz = 0

		class Z_layer:
			def __init__(self, nb_neurons, prev_nb_neurons):
				if prev_nb_neurons + nb_neurons > 0:
					np.random.seed(1)
					self.weights = np.random.uniform(-(sqrt(6) / sqrt(prev_nb_neurons + nb_neurons)),
													 +(sqrt(6) / sqrt(prev_nb_neurons + nb_neurons)),
													 [prev_nb_neurons, nb_neurons])
					# Xavier initialization
				self.biases = np.random.rand(nb_neurons, )
				self.z_vector = 0

		class A_layer:
			def __init__(self, activation_function):
				self.activation_function = activation_function
				self.a_vector = 0


	# - - - - - - - - - - - - - - - - - - - - - - - -


	# Network training function
	# Successively calls the functions feed_forward, backpropagation and update
	def train(self, X_train, y_train, test_data, epochs=1, learning_rate=0.5, mini_batch=1):

		for e in range(epochs):
			
			for i in range(len(X_train)):

				input_vector = X_train[i]
				self.feed_forward(X_train[i])				
				self.backpropagation(y_train[i])
				self.update(learning_rate, mini_batch)

				if i % 500 == 0 and i != 0:
					print(i, 'examples passed through the model')
					print("Accuracy :", self.test(test_data[0], test_data[1]))
					print(), input("press enter to continue"), print()


	# Input's forward propagation

	def feed_forward(self, input_vector):
		self.layers[0].a_layer.a_vector = input_vector

		for i in range(1, len(self.layers)):
			previous_layer = self.layers[i - 1]
			layer = self.layers[i]

			# compute z_vector
			input_vector = previous_layer.a_layer.a_vector
			weights = layer.z_layer.weights
			biases = layer.z_layer.biases
			layer.z_layer.z_vector = np.dot(input_vector, weights) + biases

			# compute a_vector
			activation_function = layer.a_layer.activation_function
			layer.a_layer.a_vector = activation_function(layer.z_layer.z_vector)

		output_layer = self.layers[-1].a_layer.a_vector

		return output_layer



	# Backpropagation

	# 1) iterates over layers in opposite way in backpropagation function
	# 2) computes gradients for each layer in output/hidden_derivation functions

	# notations
	# dE_da : Error's partial derivative over activation
	# da_dz : activation's partial derivative over preactivation
	# dz_dw : preactivation partial derivative over weights

	# opposite way iterations
	def backpropagation(self, y):
		self.output_derivation(y)

		for i in range(2, len(self.layers)):
			self.hidden_derivation(-i)

	# gradients calculation
	# - once calculated, the gradients are stocked as layers attributes
	# - functions derivatives at the end of the file
	def output_derivation(self, y):
		output_layer = self.layers[-1]
		output = output_layer.a_layer.a_vector
		dE_da = gradient_categorical_crossentropy(output, y)
		da_dz = sigmoid_derivative(output)
		dz_dw = self.layers[-2].a_layer.a_vector

		output_layer.dE_dz += dE_da * da_dz
		output_layer.update_matrix += np.dot(np.asmatrix(dz_dw).T, np.asmatrix(output_layer.dE_dz))

	def hidden_derivation(self, i):
		hidden_layer = self.layers[i]
		a_vector = hidden_layer.a_layer.a_vector

		dE_da = np.dot(self.layers[i + 1].dE_dz, self.layers[i + 1].z_layer.weights.T)
		da_dz = sigmoid_derivative(a_vector)
		dz_dw = self.layers[i - 1].a_layer.a_vector

		hidden_layer.dE_dz += dE_da * da_dz
		hidden_layer.update_matrix += np.dot(np.asmatrix(dz_dw).T, np.asmatrix(hidden_layer.dE_dz))


	# parameters update
	# called in train function
	def update(self, learning_rate, mini_batch):

		for layer in self.layers[1:]:
			layer.z_layer.weights -= (layer.update_matrix / mini_batch) * learning_rate
			layer.z_layer.biases -= (np.sum(layer.dE_dz / mini_batch)) * learning_rate
			layer.reset_update()

		

	# test function
	# prediction successive sur tout le corpus passé en arguments, renvoie le pourcentage de bonnes prédictions
	def test(self, test_X, test_y):
		good_answers = 0
		for i in range(len(test_X)):
			prediction = self.predict(test_X[i])
			true_label = test_y[i]
			if prediction == true_label:
				good_answers += 1
		return good_answers / len(test_X) * 100



	# prediction
	# returns the highest value's index in the newtork's output
	def predict(self, input_vector):
		output_vector = self.feed_forward(input_vector)
		return np.argmax(output_vector)


# functions derivatives
def sigmoid_derivative(output):
	return output * (1 - output)

def gradient_categorical_crossentropy(predictions, targets):
    return predictions - targets
