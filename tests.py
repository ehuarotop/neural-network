import click
import Utils
import pandas as pd
from NeuralNetwork import NeuralNetwork as nn
import multiprocessing
import itertools
import numpy as np

def call_cross_validation(dataset, reg_factor, n_layers, network_weights, inputs, outputs, kfolds, n_cross_val):
	Utils.cross_validation(dataset, reg_factor, n_layers, network_weights, inputs, outputs, kfolds, n_cross_val)

@click.command()
@click.option('--dataset', required=True, type=str, help='Txt file with the training set')
def main(dataset):
	reg_factors = [0.05,0.1,0.15]
	num_layers = [1,2]
	neurons_per_layer = [1,2,3]

	inputs, outputs = Utils.get_data_from_txt(dataset)

	#converting datasets to numeric from string
	inputs = inputs.apply(pd.to_numeric)
	#Performing data normalization
	inputs = Utils.apply_standard_score(inputs)

	outputs = outputs.apply(pd.to_numeric)

	procs = []

	network_confs = []

	#generating final network structure
	for num_layer in num_layers:
		if num_layer == 1:
			for i in itertools.product(neurons_per_layer):
				if dataset == 'datasets/wine_dataset.txt':
					network_confs.append([13] + list(i) + [3])
				elif dataset == 'datasets/pima_dataset.txt':
					network_confs.append([8] + list(i) + [2])
				elif dataset == 'datasets/ionosphere_dataset.txt':
					network_confs.append([34] + list(i) + [2])
		elif num_layer == 2:
			for i in itertools.product(neurons_per_layer, neurons_per_layer):
				if dataset == 'datasets/wine_dataset.txt':
					network_confs.append([13] + list(i) + [3])
				elif dataset == 'datasets/pima_dataset.txt':
					network_confs.append([8] + list(i) + [2])
				elif dataset == 'datasets/ionosphere_dataset.txt':
					network_confs.append([34] + list(i) + [2])


	#generating weights and instantiating parallel processes
	for reg_factor in reg_factors:
		for network_conf in network_confs:
			network_weights = []
			for index, layer in enumerate(network_conf):
				if index != (len(network_conf)-1):
					network_weights.append(np.random.rand(network_conf[index+1],layer+1))

			args = [dataset, reg_factor, network_conf, network_weights, inputs, outputs, 10, 1]
			args = tuple(args)
			process = multiprocessing.Process(target=call_cross_validation, args=args)
			process.start()
			procs.append(process)

	
	#Completing jobs
	for proc in procs:
		proc.join()

if __name__ == "__main__":
	main()