import numpy as np
import math

def apply_sigmoid_function(activations):
	activations = 1 / (1 + np.exp(-activations))
	return activations

class Layer:
	def __init__(self, layer_type, n_neurons, weights, activations, bias):
		self.type = layer_type
		self.n_neurons = n_neurons
		self.weights = weights
		self.activations = activations

class NeuralNetwork:
	def __init__(self, reg_factor, n_layers, initial_weights, inputs, outputs):
		
		def getInitialLayers(n_layers):
			layers = []

			for index, n_layer in enumerate(n_layers):
				if index == 0:
					layer = Layer('input', n_layer, None, None, False)
				elif index == len(n_layers) - 1:
					layer = Layer('output', n_layer, np.array(initial_weights[index-1]), None, True)
				else:
					layer = Layer('middle', n_layer, np.array(initial_weights[index-1]), None, True)

				layers.append(layer)

			return layers


		self.reg_factor = reg_factor
		self.layers = getInitialLayers(n_layers)


	def propagateInstance(self, instance):
		#Instantiating activations (plus bias) for the first layer (input layer)
		self.layers[0].activations = np.insert(np.array([instance.values]), 0, 1.0, axis=1)

		for index, layer in enumerate(self.layers):
			#Calculating propagation
			if index != 0:
				'''
				propagation is only calculated from the second layer onwards. layer.activations
				are calculated multiplying the weights of the current layer with the tranpose of the 
				activations of the previous layer (self.layers[index-1]). layer.activations are saved
				in his transpose form in order to maintain the operation fixed.
				'''
				layer.activations = np.dot(layer.weights, self.layers[index-1].activations.T).T

				layer.activations = apply_sigmoid_function(layer.activations)
				
				#adding bias to the layer just calculated (if not output layer)
				if layer.type is not 'output':
					layer.activations = np.insert(layer.activations, 0, 1.0, axis=1)

		return layer.activations 



