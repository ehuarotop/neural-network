import numpy as np
import math
import pandas as pd
import sys

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
	def __init__(self, reg_factor, n_layers, initial_weights, inputs, outputs, alpha, beta, stop_criteria, max_patience, batch_size, max_iterations, momentum, verbose):
		
		def getInitialLayers(n_layers):
			layers = []

			for index, n_layer in enumerate(n_layers):
				if index == 0:
					layer = Layer('input', n_layer, None, None, False)
				elif index == len(n_layers) - 1:
					layer = Layer('output', n_layer, initial_weights[index-1], None, True)
				else:
					layer = Layer('middle', n_layer, initial_weights[index-1], None, True)

				layers.append(layer)

			return layers

		self.network_structure = n_layers
		self.reg_factor = reg_factor
		self.layers = getInitialLayers(n_layers)
		self.inputs = inputs
		self.outputs = outputs
		self.alpha = alpha
		self.beta = beta
		self.stop = False
		self.stop_criteria = stop_criteria
		self.patience = 0
		self.max_patience = max_patience
		self.batch_size = batch_size
		self.max_iterations = max_iterations
		self.momentum = momentum
		self.verbose = verbose


	def propagateInstance(self, instance, verbose):
		#Instantiating activations (plus bias) for the first layer (input layer)
		self.layers[0].activations = np.insert(np.array([instance.values]), 0, 1.0, axis=1).T
		if verbose:
			print('\t\ta1: ', self.layers[0].activations.T[0])

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
				if verbose:
					print('\t\tz'+str(index+1)+': ', layer.activations.T[0])

				layer.activations = apply_sigmoid_function(layer.activations)

				#adding bias to the layer just calculated (if not output layer)
				if layer.type is not 'output':
					layer.activations = np.insert(layer.activations, 0, 1.0, axis=0)
					if verbose:
						print('\t\ta'+str(index+1)+': ', layer.activations.T[0])

	def calculateErrorOutputLayer(self, output_layer, instance_outputs):
		#Function to calculate error of a instance in a network with multiple outputs

		#getting activations of the output layer
		output_layer_activations = output_layer.activations 			#np.array(column vector)
		real_outputs = np.array([instance_outputs.values]).T 			#np.array(column vector)

		error = np.sum(np.add(np.multiply(-real_outputs, np.log(output_layer_activations)),
						np.multiply(-(1-real_outputs), np.log(1-output_layer_activations))))

		return error

	def backPropagateNetwork(self, meanOutputLayerError, verbose):

		self.layers[-1].deltas = meanOutputLayerError
		if verbose:
			print('\t\tdelta4: ', self.layers[-1].deltas.T[0])

		#reverse indexing excluding the output layer
		for index in range(len(self.layers)-2, 0, -1):
			#removing bias neuron for deltas in order to do multiplication when n_layer >= 4
			if(len(self.layers)) > 3 and index != (len(self.layers)-2):
				deltas_current_layer = np.array([self.layers[index+1].deltas.T[0][1:]]).T
			else:
				deltas_current_layer = self.layers[index+1].deltas

			weighted_deltas = np.dot(self.layers[index+1].weights.T, deltas_current_layer)
			layer_activations = np.multiply(self.layers[index].activations, 1 - self.layers[index].activations)
			
			self.layers[index].deltas = np.multiply(weighted_deltas, layer_activations)
			if verbose:
				print('\t\tdelta'+str(index+1)+': ', self.layers[index].deltas.T[0][1:])

		#Calculate gradients for weights
		for index,layer in enumerate(self.layers):
			if index != 0:
				layer.gradients = np.multiply(self.layers[index-1].activations, layer.deltas.T).T

				if index != len(self.layers)-1:
					layer.gradients = layer.gradients[1:]

				if verbose:
					print('\t\tGradients for Theta' + str(index))
					for gradient in layer.gradients:
						print('\t\t\t' + str(gradient))

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

	def propagateInstanceAndGetOutputLayerError(self, instance, output_layer, instance_output, verbose):
		self.propagateInstance(instance, verbose)
		return self.calculateErrorOutputLayer(self.layers[-1], instance_output)

	def getRegularized_J(self, verbose):
		############################# Gettting regularized cost ############################# 
		J = 0.0

		for index, instance in self.inputs.iterrows():
			#getting J (cost function) for the current instance
			error = self.propagateInstanceAndGetOutputLayerError(instance, self.layers[-1], self.outputs.iloc[index],verbose)

			#Adding the J (cost function) for the current instance to the output_layer_errors variable
			J += error

		#getting the mean error for each neuron on the output layer.
		J_mean = J / self.inputs.shape[0]
		regularizedError = self.getRegularizedOutputLayerError(J_mean)

		return regularizedError

	def backPropagation(self):
		num_calc_gradients = 0

		#Getting first regularized cost
		regularized_J = self.getRegularized_J(False)

		total_gradients = []
		accumulated_gradients = []

		for index, layer in enumerate(self.layers):
			if index == 0:
				total_gradients.append(None)
				accumulated_gradients.append(None)
			else:
				total_gradients.append(np.zeros((layer.weights.shape)))
				accumulated_gradients.append(np.zeros((layer.weights.shape)))

		############################# Iterating over backpropagation #############################
		for iteration in range(self.max_iterations):
			if not self.stop:
				#Printing count for current iteration
				print("Iteration # " + str(iteration + 1))

				#Getting batches
				batches = []
				n_batches = self.inputs.shape[0] // self.batch_size
				init = 0
				end = init + self.batch_size

				for i in range(n_batches+1):
					if i == n_batches:
						batch = self.inputs.iloc[init:]
					else:
						batch = self.inputs.iloc[init:end]
					
					batches.append(batch)
					init = end
					end += self.batch_size

				for batch in batches:
					for index, instance in batch.iterrows(): #self.inputs.iterrows()
						#Propagating the instance
						self.propagateInstance(instance, False)

						#getting deltas for the output layer
						delta_output_layer = np.add(self.layers[-1].activations, -np.array([self.outputs.iloc[index].values]).T)
						
						self.backPropagateNetwork(delta_output_layer, False)

						for index, layer in enumerate(self.layers):
							if index != 0:
								total_gradients[index] = np.add(total_gradients[index], layer.gradients)

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

							if self.momentum:
								#incrementing the number of gradients calculated until this moment
								num_calc_gradients += 1

								#accumulate the calculated gradients
								accumulated_gradients[index] = np.add(accumulated_gradients[index], layer.gradients)

								#Calculating the z direction based on the mean gradients and 
								z_direction = self.beta * (accumulated_gradients[index] / num_calc_gradients) + layer.gradients

								#updating weights
								layer.weights = layer.weights - self.alpha * z_direction

							else:
								#updating weights according to the calculated gradients ######UPDATING WEIGHTS
								layer.weights = layer.weights - self.alpha * layer.gradients

					#Getting the new regularized error after updating weights
					new_regularized_J = self.getRegularized_J(False)

					if new_regularized_J - regularized_J < self.stop_criteria:
						self.patience += 1
						regularized_J = new_regularized_J
						if self.patience == self.max_patience:
							self.stop = True
					else:
						self.patience = 0
				

	def predict(self, instances):
		predictions = []

		#Receives a set of instances to be predicted
		for index, instance in instances.iterrows():
			self.propagateInstance(instance, False)
			predicted_probabilities = self.layers[-1].activations

			maximum = 0
			max_index = 0
			
			for index, predicted_proba in enumerate(predicted_probabilities):
				if predicted_proba[0] > maximum:
					maximum = predicted_proba[0]
					max_index = index

			prediction = []
			for index, predicted_proba in enumerate(predicted_probabilities):
				if index == max_index:
					predicted_proba[0] = 1.0
				else:
					predicted_proba[0] = 0.0

				prediction.append(predicted_proba[0])

			predictions.append(prediction)

		#returning predictions as a dataframe considering the indices of the instances
		return pd.DataFrame(predictions).set_index(instances.index)

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
				regularized_J_plus_eps = self.getRegularized_J(False)

				weights[row][col] = current_weight - epsilon
				regularized_J_minus_eps = self.getRegularized_J(False)

				numeric_gradients[row][col] = (regularized_J_plus_eps - regularized_J_minus_eps) / (2*epsilon)

				weights[row][col] = weights_copy[row][col]

		return numeric_gradients

	def simplebackPropagationForNumericVerification(self):

		def printNetworkWeights():
			for index, layer in enumerate(self.layers):
				if index != 0:
					print('Tethas 0' + str(index))
					print('')
					print(layer.weights)
					print('')

		def printTrainingSet():
			for index, instance in self.inputs.iterrows():
				print('Example ' + str(index+1))
				print('x: ', list(instance))
				print('y: ', list(self.outputs.iloc[index]))
				print('')

		print('Numerical Verification for Backpropagation Algorithm')
		print('=====================================================')

		print('Regularization parameter ("λ"): ' + str(self.reg_factor))
		print('Network Layers structure: ' + str(self.network_structure))
		print('')

		#Printing weights for each layer (except for input layer)
		printNetworkWeights()

		print('Training Set')
		printTrainingSet()

		print('------------------------------------------------------')
		print('Calculating network error J')

		total_gradients = []

		for index, layer in enumerate(self.layers):
			if index == 0:
				total_gradients.append(None)
			else:
				total_gradients.append(np.zeros((layer.weights.shape)))

		############################# Iterating over backpropagation #############################
		for index, instance in self.inputs.iterrows():
			print('\tProcessing example # ' + str(index+1))

			#Propagating the instance
			print('\tPropagating instance ' + str(list(instance)))
			#self.propagateInstance(instance, True)
			J_example = self.propagateInstanceAndGetOutputLayerError(instance, self.layers[-1], self.outputs.iloc[index],True)

			print('\tPredicted output Example: ' + str(index+1), self.layers[-1].activations.T[0])
			print('\tExpected output Example : ' + str(index+1), np.array([self.outputs.iloc[index].values][0]))
			print('\tJ for Example ' + str(index+1) + ': ', J_example)

			#getting deltas for the output layer
			delta_output_layer = np.add(self.layers[-1].activations, -np.array([self.outputs.iloc[index].values]).T)
			
			#print('Executing Backpropagation for Example ' + str(index+1))
			print('\tCalculating gradients based on example ' + str(index+1))
			self.backPropagateNetwork(delta_output_layer, True)

			for index, layer in enumerate(self.layers):
				if index != 0:
					total_gradients[index] = np.add(total_gradients[index], layer.gradients)

		print('\nDataset processed completely. Calculating regularized gradients\n')

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

				print('\tFinal gradients for Theta' + str(index) + ' (with regularization)')
				for gradient in layer.gradients:
					print('\t\t' + str(gradient))
				print('\tGradients calculated numerically for Theta' + str(index))
				for num_gradient in numerical_gradients:
					print('\t\t' + str(num_gradient))

		new_regularized_J = self.getRegularized_J(False)
		print('J total do dataset', new_regularized_J)

