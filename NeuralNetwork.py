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
		self.alpha = 0.0
		self.stop = False
		self.stop_criteria = 0.0005


	def propagateInstance(self, instance):
		#Instantiating activations (plus bias) for the first layer (input layer)
		self.layers[0].activations = np.insert(np.array([instance.values]), 0, 1.0, axis=1).T

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

		#return self.layers[-1].activations

	def calculateErrorOutputLayer(self, output_layer, instance_outputs):
		#Function to calculate error of a instance in a network with multiple outputs

		#getting activations of the output layer
		output_layer_activations = output_layer.activations 			#np.array(column vector)
		real_outputs = np.array([instance_outputs.values]).T 			#np.array(column vector)

		error = np.sum(np.add(np.multiply(-real_outputs, np.log(output_layer_activations)),
						np.multiply(-(1-real_outputs), np.log(1-output_layer_activations))))

		return error

	def backPropagateNetwork(self, meanOutputLayerError):

		self.layers[-1].deltas = meanOutputLayerError

		#reverse indexing excluding the output layer
		for index in range(len(self.layers)-2, 0, -1):
			#removing bias neuron for deltas in order to do multiplication when n_layer >= 4
			if(len(self.layers)) > 3 and index != (len(self.layers)-2):
				deltas_current_layer = np.array([self.layers[index+1].deltas.T[0][1:]]).T
			else:
				deltas_current_layer = self.layers[index+1].deltas

			#weighted_deltas = np.dot(self.layers[index+1].weights.T, self.layers[index+1].deltas)
			weighted_deltas = np.dot(self.layers[index+1].weights.T, deltas_current_layer)
			layer_activations = np.multiply(self.layers[index].activations, 1 - self.layers[index].activations)
			
			self.layers[index].deltas = np.multiply(weighted_deltas, layer_activations)
			print('deltas-layer' + str(index+1), self.layers[index].deltas)

		#Calculate gradients for weights
		for index,layer in enumerate(self.layers):
			if index != 0:
				layer.gradients = np.multiply(self.layers[index-1].activations, layer.deltas.T).T

				if index != len(self.layers)-1:
					layer.gradients = layer.gradients[1:]

				print('gradients' + str(index+1) , layer.gradients)

	def getRegularizedOutputLayerError(self, meanOutputLayerError):
		regularizedError = 0.0
		
		for index, layer in enumerate(self.layers):
			if index != 0:
				#Setting bias element to zero
				weights = np.copy(layer.weights)
				weights[:,0] = 0

				#accumulating errors in variable regularizedError
				regularizedError += np.sum(np.square(weights))

		regularizedError = regularizedError*self.reg_factor/(2*self.inputs.shape[0])

		regularizedError += meanOutputLayerError

		return regularizedError

	def propagateInstanceAndGetOutputLayerError(self, instance, output_layer, instance_output):
		self.propagateInstance(instance)
		return self.calculateErrorOutputLayer(self.layers[-1], instance_output)

	def getRegularized_J(self):
		############################# Gettting regularized cost ############################# 
		J = 0.0

		for index, instance in self.inputs.iterrows():
			#getting J (cost function) for the current instance
			error = self.propagateInstanceAndGetOutputLayerError(instance, self.layers[-1], self.outputs.iloc[index])
			print('erro J ' + str(index+1), error)

			#Adding the J (cost function) for the current instance to the output_layer_errors variable
			J += error

		#getting the mean error for each neuron on the output layer.
		J_mean = J / self.inputs.shape[0]
		regularizedError = self.getRegularizedOutputLayerError(J_mean)

		return regularizedError

	def backPropagation(self):

		#Getting first regularized cost
		#regularized_J = self.getRegularized_J()

		total_gradients = []

		for index, layer in enumerate(self.layers):
			if index == 0:
				total_gradients.append(None)
			else:
				total_gradients.append(np.zeros((layer.weights.shape)))

		############################# Iterating over backpropagation #############################
		for iteration in range(1):
			#if not self.stop:
			#Printing count for current iteration
			print("Iteration # " + str(iteration + 1))

			for index, instance in self.inputs.iterrows():
				#Propagating the instance
				self.propagateInstance(instance)
				print('propagation ' + str(index+1), self.layers[-1].activations)

				print('saida esperada' + str(index+1), np.array([self.outputs.iloc[index].values]).T)

				#getting deltas for the output layer
				delta_output_layer = np.add(self.layers[-1].activations, -np.array([self.outputs.iloc[index].values]).T)
				print('erro na saida' + str(index+1), delta_output_layer)
				
				self.backPropagateNetwork(delta_output_layer)

				for index, layer in enumerate(self.layers):
					if index != 0:
						total_gradients[index] = np.add(total_gradients[index], layer.gradients)

				print('------------------------------------------------------')

			print('------------------ Dataset Completado ------------------')

			for index, layer in enumerate(self.layers):
				if index != 0:
					#Getting a copy of the weights for the current layer
					weights = np.copy(layer.weights)
					#setting bias column to zero (regularization does not consider bias)
					weights[:,0] = 0
					#applying regularization factor over the weights array
					weights = weights*self.reg_factor

					#getting regularized gradients (from total_gradients):
					regularized_gradients = np.add(total_gradients[index], weights) / self.inputs.shape[0]

					#assigning the regularized gradients to layer.gradients
					layer.gradients = regularized_gradients

					print('regularized gradients' + str(index+1), regularized_gradients)

					#updating weights according to the calculated gradients
					#layer.weights = layer.weights - self.alpha * layer.gradients

			new_regularized_J = self.getRegularized_J()
			print('J total do dataset', new_regularized_J)

				#print(regularized_J, new_regularized_J)

				#if regularized_J - new_regularized_J < self.stop_criteria:
				#	self.stop = True
				#else:
				#	regularized_J = new_regularized_J


	def numerical_verification(self, weights, epsilon):
		#Getting a copy of the weights for the current layer
		weights_copy = np.copy(weights)
		
		#setting bias column to zero (regularization does not consider bias)
		#weights[:,0] = 0

		rows, cols = weights.shape

		numeric_gradients = np.zeros(weights.shape)

		for row in range(rows):
			for col in range(cols):
				current_weight = weights[row][col]

				weights[row][col] = current_weight + epsilon
				regularized_J_plus_eps = self.getRegularized_J()

				weights[row][col] = current_weight - epsilon
				regularized_J_minus_eps = self.getRegularized_J()

				numeric_gradients[row][col] = (regularized_J_plus_eps - regularized_J_minus_eps) / (2*epsilon)

				weights[row][col] = weights_copy[row][col]

		return numeric_gradients


	def simplebackPropagationForNumericVerification(self):

		total_gradients = []

		for index, layer in enumerate(self.layers):
			if index == 0:
				total_gradients.append(None)
			else:
				total_gradients.append(np.zeros((layer.weights.shape)))

		############################# Iterating over backpropagation #############################
		for index, instance in self.inputs.iterrows():
			#Propagating the instance
			self.propagateInstance(instance)
			print('propagation ' + str(index+1), self.layers[-1].activations)

			print('saida esperada' + str(index+1), np.array([self.outputs.iloc[index].values]).T)

			#getting deltas for the output layer
			delta_output_layer = np.add(self.layers[-1].activations, -np.array([self.outputs.iloc[index].values]).T)
			print('erro na saida' + str(index+1), delta_output_layer)
			
			self.backPropagateNetwork(delta_output_layer)

			for index, layer in enumerate(self.layers):
				if index != 0:
					total_gradients[index] = np.add(total_gradients[index], layer.gradients)

			print('------------------------------------------------------')

		print('------------------ Dataset Completado ------------------')

		for index, layer in enumerate(self.layers):
			if index != 0:
				#performing numerical verification
				numerical_gradients = self.numerical_verification(layer.weights, 0.00001)

				#Getting a copy of the weights for the current layer
				weights = np.copy(layer.weights)
				
				#setting bias column to zero (regularization does not consider bias)
				weights[:,0] = 0

				#applying regularization factor over the weights array
				weights = weights*self.reg_factor

				#getting regularized gradients (from total_gradients):
				regularized_gradients = np.add(total_gradients[index], weights) / self.inputs.shape[0]

				#assigning the regularized gradients to layer.gradients
				layer.gradients = regularized_gradients

				print('regularized gradients' + str(index+1), regularized_gradients)
				print('numerical gradients' + str(index+1), numerical_gradients)

				#updating weights according to the calculated gradients
				#layer.weights = layer.weights - self.alpha * layer.gradients

		new_regularized_J = self.getRegularized_J()
		print('J total do dataset', new_regularized_J)

