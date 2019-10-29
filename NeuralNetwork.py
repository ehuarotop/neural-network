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
		self.deltas = np.zeros((self.n_neurons,1))

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
		self.inputs = inputs
		self.outputs = outputs
		self.alpha = 0.15


	def propagateInstance(self, instance):
		#Instantiating activations (plus bias) for the first layer (input layer)
		self.layers[0].activations = np.insert(np.array([instance.values]), 0, 1.0, axis=1).T
		#print(self.layers[0].activations.shape)

		for index, layer in enumerate(self.layers):
			#Calculating propagation
			if index != 0:
				'''
				propagation is only calculated from the second layer onwards. layer.activations
				are calculated multiplying the weights of the current layer with the tranpose of the 
				activations of the previous layer (self.layers[index-1]). layer.activations are saved
				in his transpose form in order to maintain the operation fixed.
				'''
				layer.activations = np.dot(layer.weights, self.layers[index-1].activations)

				layer.activations = apply_sigmoid_function(layer.activations)

				#adding bias to the layer just calculated (if not output layer)
				if layer.type is not 'output':
					layer.activations = np.insert(layer.activations, 0, 1.0, axis=0)

		return self.layers[-1].activations

	def calculateErrorOutputLayer(self, output_layer, instance_outputs):
		#Function to calculate error of a instance in a network with multiple outputs

		#getting activations of the output layer
		output_layer_activations = output_layer.activations 			#np.array(column vector)
		real_outputs = np.array([instance_outputs.values]).T 			#np.array(column vector)

		error = np.sum(np.add(np.multiply(-real_outputs, np.log(output_layer_activations)),
						np.multiply(-(1-real_outputs), np.log(1-output_layer_activations))))

		return error

	def backPropagateInstance(self, instance, outputs):
		#Getting outputs in the correct format used in this implementation
		outputs = np.array([outputs.values]).T

		#Assigning deltas to output layer
		self.layers[-1].deltas = np.add(self.layers[-1].activations, -outputs)

		#reverse indexing excluding the output layer
		for index in range(len(self.layers)-2, 0, -1):
			weighted_deltas = np.dot(self.layers[index+1].weights.T, self.layers[index+1].deltas)
			layer_activations = np.multiply(self.layers[index].activations, 1 - self.layers[index].activations)
			
			self.layers[index].deltas = np.multiply(weighted_deltas, layer_activations)

		#Calculate gradients for weights
		for index,layer in enumerate(self.layers):
			if index != 0:
				layer.gradients = np.multiply(self.layers[index-1].activations, layer.deltas.T).T
				#print(layer.weights.shape)
				#print(layer.gradients.shape)

				if index != len(self.layers)-1:
					layer.gradients = layer.gradients[1:]


				#updating weights
				layer.weights = layer.weights - self.alpha * layer.gradients

	def propagateInstanceAndGetOutputLayerError(self, instance, output_layer, instance_output):
		self.propagateInstance(instance)
		return self.calculateErrorOutputLayer(self.layers[-1], instance_output)

	def backPropagation(self):
		meanError = 1000000000
		print(meanError)
		while meanError > 0.05:
			totalError = 0.0
			for index, instance in self.inputs.iterrows():
				#propagating the instance
				self.propagateInstance(instance)
				#backpropagating the instance and updating the weights
				self.backPropagateInstance(instance, self.outputs.iloc[index])

				#getting J (total error)
				totalError += self.propagateInstanceAndGetOutputLayerError(instance, self.layers[-1], self.outputs.iloc[index])

			meanError = totalError / self.inputs.shape[0]
			print(meanError)