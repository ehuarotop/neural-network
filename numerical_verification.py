import click
import Utils
import pandas as pd
from NeuralNetwork import NeuralNetwork as nn

@click.command()
@click.option('--network', required=True, type=str, help='Txt File indicating the network structure')
@click.option('--initial_weights', required=True, type=str, help='Txt File indicating the initial weights for the network')
@click.option('--dataset', required=True, type=str, help='Txt file with the training set')
def main(network, initial_weights, dataset):
	#parsing network.txt file to variables with the correct data type
	reg_factor, n_layers = Utils.parser_network_file(network)

	#parsing initial_weights.txt file to variables with the correct datatype. Last element of this list will be the output neuron
	network_weights = Utils.parser_initial_weights_file(initial_weights)

	inputs, outputs = Utils.get_data_from_txt(dataset)

	#converting datasets to numeric from string
	inputs = inputs.apply(pd.to_numeric)
	outputs = outputs.apply(pd.to_numeric)

	neural_network = nn(reg_factor, n_layers, network_weights, inputs, outputs,
									0.05, 0.9, 0.000005, 25, 50, 200, True, False)


	neural_network.simplebackPropagationForNumericVerification()

if __name__ == "__main__":
	main()