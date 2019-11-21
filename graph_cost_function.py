import click
import Utils
import pandas as pd
from NeuralNetwork import NeuralNetwork as nn
from matplotlib import pyplot as plt
import os
import numpy as np

@click.command()
@click.option('--network', required=True, type=str, help='Txt File indicating the network structure')
@click.option('--initial_weights', required=True, type=str, help='Txt File indicating the initial weights for the network')
@click.option('--dataset', required=True, type=str, help='Txt file with the training set')
@click.option('--batch_size', required=True, type=str, help='Size of the batch to generate curve')
def main(network, initial_weights, dataset, batch_size):
	#parsing network.txt file to variables with the correct data type
	reg_factor, n_layers = Utils.parser_network_file(network)

	#parsing initial_weights.txt file to variables with the correct datatype. Last element of this list will be the output neuron
	network_weights = Utils.parser_initial_weights_file(initial_weights)

	inputs, outputs = Utils.get_data_from_txt(dataset)

	#converting datasets to numeric from string
	inputs = inputs.apply(pd.to_numeric)
	#Performing data normalization
	inputs = Utils.apply_standard_score(inputs)

	outputs = outputs.apply(pd.to_numeric)

	#Instantiating Neural Network object
	neural_network = nn(reg_factor, n_layers, network_weights, 
							inputs, 
							outputs,
							0.10, 0.9, 0.000005, 800, 50, 200, True, False)

	J_values = neural_network.graphCostFunction(int(batch_size))

	x = np.array([str(int(batch_size)*x) for x in range(1, len(J_values) + 1)])

	plt.figure(figsize=(10,6))
	plt.title('Gráfico da Função de custo J - ' + batch_size + ' instâncias')
	plt.ylabel('J value')
	plt.xlabel('# Instância')
	plt.plot(x, np.array(J_values), '-o')
	plt.savefig('cost_function_' + os.path.basename(dataset) + '.png')
	plt.close()

	#Utils.cross_validation(dataset, reg_factor, n_layers, network_weights, inputs, outputs, 10, 1)

if __name__ == "__main__":
	main()