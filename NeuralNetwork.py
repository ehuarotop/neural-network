import numpy as np

class Layer:
	def __init__(self, layer_type, n_neurons, weights, activations):
		self.type = layer_type
		
		if self.type is not 'input':
			#preprocessing weights
			for weight in weights:
				weight = np.insert(np.array(weight), 0, 1.0, axis=0)

		self.n_neurons = n_neurons
		self.weights = weights
		self.activations = activations

class NeuralNetwork:
	def __init__(self, reg_factor, n_layers, initial_weights, inputs, outputs):
		
		def getInitialLayers(n_layers):
			layers = []

			for index, n_layer in enumerate(n_layers):
				if index == 0:
					layer = Layer('input', n_layer, None, None)
				elif index == len(n_layers) - 1:
					layer = Layer('output', n_layer, initial_weights[index-1], None)
				else:
					layer = Layer('middle', n_layer, initial_weights[index-1], None)

				layers.append(layer)

			return layers


		self.reg_factor = reg_factor
		self.layers = getInitialLayers(n_layers)

		print(self.layers)