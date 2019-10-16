import pandas as pd
import json
import numpy as np
import csv
import os

def get_data_from_csv(csv_file, sep):
	#reading dataset from csv
	data = pd.read_csv(csv_file, sep)

	#reading dataset descritpion from json file.
	#with open(csv_file[:-4] + "-description.json") as desc_file:
	#	data_desc = json.loads(desc_file.read())

	return data, data_desc

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
	n_inputs = int(lines[1][0])
	n_neurons = int(lines[2][0])
	n_outputs = int(lines[3][0])

	return reg_factor, n_inputs, n_neurons, n_outputs


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

	return network_weights

def write_line_to_csv(csv_file, list_values):
	with open(csv_file, mode='a') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(list_values)

def hasHeader(csv_file):
	with open(csv_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			if 'cross_val' in row:
				return True
			else:
				return False

			break

def cross_validation(dataset_name, data, data_desc, n_trees, kfolds, n_cross_val=1):
	total_accuracies = []
	total_F1s = []

	#Getting the class column (assuming class column in the last column) 
	classe = list(data.columns.values)[-1]

	#Getting actual class instances
	class_instances = data[classe]

	#Getting unique class instances values from class column
	unique_class_values = list(class_instances.unique())

	#ordering the class values
	unique_class_values.sort()

	if not os.path.exists(dataset_name[:-4] + '-metrics.csv'):
		#Bulding csv file headers line for any number of classes
		headers_csv = ['cross_val', 'kfold', 'n_trees','accuracy']
		headers_csv += ['class_' + str(unique_class) + '_recall' for unique_class in unique_class_values]
		headers_csv += ['mean_recall']
		headers_csv += ['class_' + str(unique_class) + '_precision' for unique_class in unique_class_values]
		headers_csv += ['mean_precision']
		headers_csv += ['class_' + str(unique_class) + '_F1' for unique_class in unique_class_values]
		headers_csv += ['mean_F1']

		write_line_to_csv(dataset_name[:-4] + '-metrics.csv', headers_csv)

	for cross_val in range(n_cross_val):
		print("Cross validation # " + str(cross_val+1))
		print("----------------------------")
		accuracies = []
		F1s_classes = []

		#Reordering data randomly
		data = data.reindex(np.random.permutation(data.index))

		data_classes = []

		for class_value in unique_class_values:
			data_class = data[data[classe] == class_value]
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

				test_data = pd.concat([test_data, test_data_class]).reset_index(drop=True)
				training_data = pd.concat([training_data, training_data_class]).reset_index(drop=True)

			#Getting random forest model
			rforest = rf(n_trees,training_data, data_desc)

			#fit the model
			rforest.fit()

			#getting predictions from the random forest.
			predictions = rforest.predict(test_data)

			#Getting confusion matrix
			cf = getConfusionMatrix(predictions, test_data, unique_class_values)

			#Getting some metrics from the confusion matrix to validate the model
			accuracy, recalls, precisions, F1s = calcMetrics(cf, unique_class_values)

			#Concatenating metrics into a list to be exported to a csv file
			list_of_metric_values = [cross_val+1, kfold+1, n_trees] + [accuracy] + recalls + [np.mean(recalls)] + precisions + [np.mean(precisions)] + F1s + [np.mean(F1s)]
			
			#Writing list of metrics computed for the current fold to a csv file.
			write_line_to_csv(dataset_name[:-4] + '-metrics.csv', list_of_metric_values)

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


def getConfusionMatrix(predictions, test_data, unique_class_values):
	confusion_matrix = np.zeros((len(unique_class_values), len(unique_class_values)), dtype=int)

	#ordering the class values
	unique_class_values.sort()

	for index, row in test_data.iterrows():
		x_val = -1
		y_val = -1

		for index2, value in enumerate(unique_class_values):
			if isinstance(row[-1], np.float64):
				test_data_class = np.int64(row[-1])
			else:
				test_data_class = row[-1]

			if predictions[index] == value:
				x_val = index2
			if test_data_class == value:
				y_val = index2

		confusion_matrix[x_val][y_val] += 1

	return confusion_matrix

def calcMetrics(confusion_matrix, unique_class_values):
	#Calculating accuracy
	accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)

	#Calculating recall
	recalls = []
	for index, value in enumerate(unique_class_values):
		row = confusion_matrix[index, :]
		recall = confusion_matrix[index][index] / row.sum() if row.sum() else 0
		recalls.append(recall)

	#Calculating precision
	precisions = []
	for index, value in enumerate(unique_class_values):
		column = confusion_matrix[:,index]
		precision = confusion_matrix[index][index] / column.sum() if column.sum() else 0
		precisions.append(precision)

	#Calculating F1-measure
	F1s = []
	for index, value in enumerate(unique_class_values):
		F1 = 2 * (precisions[index] * recalls[index]) / (precisions[index] + recalls[index]) if (precisions[index] + recalls[index]) else 0
		F1s.append(F1)

	return accuracy, recalls, precisions, F1s