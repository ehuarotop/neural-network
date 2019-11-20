import pandas as pd
import json
import numpy as np
import csv
import os
from NeuralNetwork import NeuralNetwork as nn

def get_data_from_txt(csv_file):
	#reading dataset from csv
	#data = pd.read_csv(csv_file, sep)
	lines = open(csv_file,'r').readlines()

	inputs = []
	outputs = []

	for line in lines:
		line = line[:-1]
		line_parts = line.split(';')
		instance, output = line_parts[0], line_parts[1]

		inputs.append(instance.split(','))
		outputs.append(output.split(','))

	return pd.DataFrame(inputs), pd.DataFrame(outputs)

def read_file(csv_file, delimiter=None):
	lines = []
	with open(csv_file) as csv_file:
		if delimiter is None:
			csv_reader = csv.reader(csv_file, delimiter=',')
		else:
			csv_reader = csv.reader(csv_file, delimiter=delimiter)
		for row in csv_reader:
			lines.append(row)

	return lines

def parser_network_file(network_file):
	#reading lines from the file
	lines = read_file(network_file)

	#assigning the values to corresponding variables with the correct datatypes
	reg_factor = float(lines[0][0])

	n_layers = []
	
	for layer in lines[1:]:
		n_layer = int(layer[0])	
		n_layers.append(n_layer)

	return reg_factor, n_layers


def parser_initial_weights_file(initial_weights_file):
	#reading lines from the file
	layers = read_file(initial_weights_file, ';')

	network_weights = []
	
	#Iterating over the lines (layers) to catch weights for each neuron
	for layer in layers:
		layer_weights = []
		for neuron in layer:
			#Splitting each neuron and converting it from string to float
			neuron_weights = neuron.split(',')
			neuron_weights = [float(weight) for weight in neuron_weights]
			#Saving the weights in a new list for each layer
			layer_weights.append(neuron_weights)

		#saving weights for each layer
		network_weights.append(layer_weights)

	#converting network_weights from list of list to 2D np.array 
	network_weights = np.array([np.array(xi) for xi in network_weights])

	return network_weights

def format_datasets(csv_files):
	#removing any pre-existent txt files in datasets directory
	os.system('rm ./datasets/*.txt')

	for csv_file in csv_files:
		if csv_file == 'datasets/wine.data':
			#wine dataset

			#reading lines from original csv file
			lines = read_file(csv_file, delimiter=',')

			#processing each line and obtained separated file for attributes and outputs.
			for line in lines:
				write_line_to_csv('datasets/wine_dataset_instances.txt',line[1:], 'a')
				if line[0] == '1':
					write_line_to_csv('datasets/wine_dataset_classes.txt', ['1.0','0.0','0.0'], 'a')
				elif line[0] == '2':
					write_line_to_csv('datasets/wine_dataset_classes.txt', ['0.0','1.0','0.0'], 'a')
				elif line[0] == '3':
					write_line_to_csv('datasets/wine_dataset_classes.txt', ['0.0','0.0', '1.0'], 'a')

			#Combining attributes and outputs files in the desired format.
			a=open('datasets/wine_dataset_instances.txt','r').readlines()
			b=open('datasets/wine_dataset_classes.txt','r').readlines()
			
			with open('datasets/wine_dataset.txt','w') as out:
			    for i in range(len(a)):
			    	out.write(a[i].rstrip() + ';' + b[i])

			#removing temporary files.
			os.system('rm ./datasets/wine_dataset_instances.txt')
			os.system('rm ./datasets/wine_dataset_classes.txt')

		if csv_file == 'datasets/pima.tsv':
			#pima dataset

			#reading lines from the original tsv file
			lines = read_file(csv_file, delimiter='	')

			#removing the first line (headers)
			lines = lines[1:]

			for line in lines:
				write_line_to_csv('datasets/pima_dataset_instances.txt',line[:-1], 'a')
				
				if line[-1] == '1':
					write_line_to_csv('datasets/pima_dataset_classes.txt', ['1.0','0.0'], 'a')
				elif line[-1] == '0':
					write_line_to_csv('datasets/pima_dataset_classes.txt', ['0.0','1.0'], 'a')

			#Combining attributes and outputs files in the desired format.
			a=open('datasets/pima_dataset_instances.txt','r').readlines()
			b=open('datasets/pima_dataset_classes.txt','r').readlines()
			
			with open('datasets/pima_dataset.txt','w') as out:
			    for i in range(len(a)):
			    	out.write(a[i].rstrip() + ';' + b[i])

			#removing temporary files.
			os.system('rm ./datasets/pima_dataset_instances.txt')
			os.system('rm ./datasets/pima_dataset_classes.txt')

		if csv_file == 'datasets/ionosphere.data':
			#wine dataset

			#reading lines from original csv file
			lines = read_file(csv_file, delimiter=',')

			#processing each line and obtained separated file for attributes and outputs.
			for line in lines:
				write_line_to_csv('datasets/ionosphere_dataset_instances.txt',line[:-1], 'a')
				if line[-1] == 'g':
					write_line_to_csv('datasets/ionosphere_dataset_classes.txt', ['1.0','0.0'], 'a')
				elif line[-1] == 'b':
					write_line_to_csv('datasets/ionosphere_dataset_classes.txt', ['0.0','1.0'], 'a')

			#Combining attributes and outputs files in the desired format.
			a=open('datasets/ionosphere_dataset_instances.txt','r').readlines()
			b=open('datasets/ionosphere_dataset_classes.txt','r').readlines()
			
			with open('datasets/ionosphere_dataset.txt','w') as out:
			    for i in range(len(a)):
			    	out.write(a[i].rstrip() + ';' + b[i])

			#removing temporary files.
			os.system('rm ./datasets/ionosphere_dataset_instances.txt')
			os.system('rm ./datasets/ionosphere_dataset_classes.txt')


def write_line_to_csv(csv_file, list_values, mode):
	with open(csv_file, mode=mode) as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(list_values)

def apply_standard_score(df):
	for column in df:
		df[column] = (df[column] - df[column].mean())/(df[column].std(ddof=0)+ 0.000001)

	return df

def cross_validation(dataset_name, reg_factor, n_layers, network_weights, inputs,outputs, kfolds, n_cross_val=1):
	total_accuracies = []
	total_F1s = []

	if dataset_name == 'datasets/wine_dataset.txt':
		unique_class_values = ['1','2','3']
	elif dataset_name == 'datasets/ionosphere_dataset.txt':
		unique_class_values = ['good', 'bad']
	elif dataset_name == 'datasets/pima_dataset.txt':
		unique_class_values = ['0','1']

	csv_file = dataset_name[:-4] + '_' + str(reg_factor) + '_' + str(n_layers) + '-metrics.csv'

	if not os.path.exists(csv_file):
		#Bulding csv file headers line for any number of classes
		headers_csv = ['cross_val', 'kfold','accuracy']
		headers_csv += ['class_' + str(unique_class) + '_recall' for unique_class in unique_class_values]
		headers_csv += ['mean_recall']
		headers_csv += ['class_' + str(unique_class) + '_precision' for unique_class in unique_class_values]
		headers_csv += ['mean_precision']
		headers_csv += ['class_' + str(unique_class) + '_F1' for unique_class in unique_class_values]
		headers_csv += ['mean_F1']

		write_line_to_csv(csv_file, headers_csv, 'w')

	for cross_val in range(n_cross_val):
		print("Cross validation # " + str(cross_val+1))
		print("----------------------------")
		accuracies = []
		F1s_classes = []

		#Reordering data randomly (outputs not needed because it will be accesed by index)
		inputs = inputs.reindex(np.random.permutation(inputs.index))

		#Getting unique outputs in the current dataset (resetting index and dropping the old index)
		unique_outputs = outputs.drop_duplicates().reset_index(drop=True)

		#List containing the separated dataframes for each class.
		data_classes = []

		#Iterating over the outputs to separate the inputs in classes.
		for index, unique_output in unique_outputs.iterrows():
			class_indexes = []
			for index2,output in outputs.iterrows():
				#Verifying if the rows are equal between output and unique_output. (3 classes)
				if unique_output.shape[0] == 3 and unique_output.eq(output).drop_duplicates().shape[0] == 1:
					class_indexes.append(index2)
				#Verification for 2 classes
				elif unique_output.shape[0] == 2 and unique_output.eq(output).drop_duplicates().iloc[0] == True:
					class_indexes.append(index2)

			#Appending part of the dataframe for the corresponding class_indexes to data_classes
			data_class = inputs.iloc[class_indexes]
			data_classes.append(data_class)

		for kfold in range(kfolds):
			print("Working on kfold " + str(kfold+1) + " of " + str(kfolds))
			test_data = pd.DataFrame.from_records([])
			training_data = pd.DataFrame.from_records([])

			for data_class in data_classes:
				#Splitting data into 'kfolds' folds
				splitted_data_class = np.array_split(data_class,kfolds)

				test_data_class = splitted_data_class[kfold]
				training_data_class = pd.DataFrame.from_records([])

				for i in range(kfolds):
					if i != kfold:
						training_data_class = pd.concat([training_data_class, splitted_data_class[i]])

				test_data = pd.concat([test_data, test_data_class])#.reset_index(drop=True)
				training_data = pd.concat([training_data, training_data_class])#.reset_index(drop=True)

			#Instantiating Neural Network object
			neural_network = nn(reg_factor, n_layers, network_weights, 
									training_data.reset_index(drop=True), 
									outputs.iloc[list(training_data.index)].reset_index(drop=True),
									0.10, 0.9, 0.000005, 800, 50, 200, True, False)

			#Fitting the neural network model
			neural_network.backPropagation()

			#Performing actual predictions
			predictions = neural_network.predict(test_data)

			#Getting confusion matrix
			cf = getConfusionMatrix(predictions, test_data, outputs, unique_outputs)

			#print(cf)

			#Getting some metrics from the confusion matrix to validate the model
			accuracy, recalls, precisions, F1s = calcMetrics(cf, unique_outputs)

			#Concatenating metrics into a list to be exported to a csv file
			list_of_metric_values = [cross_val+1, kfold+1] + [accuracy] + recalls + [np.mean(recalls)] + precisions + [np.mean(precisions)] + F1s + [np.mean(F1s)]
			
			#Writing list of metrics computed for the current fold to a csv file.
			write_line_to_csv(csv_file, list_of_metric_values, 'a')

			#Collecting accuracies in order to show this information after cross validation execution.
			accuracies.append(accuracy)
			F1s_classes.append(np.mean(F1s))
			total_accuracies.append(accuracy)
			total_F1s.append(np.mean(F1s))

		print("Accuracy: " + str(np.mean(accuracies)) + " ± " + str(np.std(accuracies)))
		print("F1 Measure: " + str(np.mean(F1s_classes)) + " ± " + str(np.std(F1s_classes)))

	print("---------------------------------------------")
	print("Total Accuracy: " + str(np.mean(total_accuracies)) + " ± " + str(np.std(total_accuracies)))
	print("Total F1-measure: " + str(np.mean(total_F1s)) + " ± " + str(np.std(total_F1s)))


def getConfusionMatrix(predictions, test_data, outputs, unique_outputs):
	confusion_matrix = np.zeros((len(unique_outputs), len(unique_outputs)), dtype=int)

	x_val = -1
	y_val = -1
	
	for index,row in predictions.iterrows():
		for index2, unique_output in unique_outputs.iterrows():
			if unique_output.shape[0] == 3:
				if unique_output.eq(row).drop_duplicates().shape[0] == 1:
					x_val = index2
				if unique_output.eq(outputs.iloc[index]).drop_duplicates().shape[0] == 1:
					y_val = index2
			elif unique_output.shape[0] == 2:
				if unique_output.eq(row).drop_duplicates().iloc[0] == True:
					x_val = index2
				if unique_output.eq(outputs.iloc[index]).drop_duplicates().iloc[0] == True:
					y_val = index2

		confusion_matrix[x_val][y_val] += 1

	return confusion_matrix

def calcMetrics(confusion_matrix, unique_outputs):
	#Calculating accuracy
	accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)

	#Calculating recall
	recalls = []
	for index, value in enumerate(unique_outputs):
		row = confusion_matrix[index, :]
		recall = confusion_matrix[index][index] / row.sum() if row.sum() else 0
		recalls.append(recall)

	#Calculating precision
	precisions = []
	for index, value in enumerate(unique_outputs):
		column = confusion_matrix[:,index]
		precision = confusion_matrix[index][index] / column.sum() if column.sum() else 0
		precisions.append(precision)

	#Calculating F1-measure
	F1s = []
	for index, value in enumerate(unique_outputs):
		F1 = 2 * (precisions[index] * recalls[index]) / (precisions[index] + recalls[index]) if (precisions[index] + recalls[index]) else 0
		F1s.append(F1)

	return accuracy, recalls, precisions, F1s
