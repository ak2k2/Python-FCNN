import numpy as np
import pandas as pd

from old.fcnn import initialize_network, predict, train_network
from helpers import (
    compute_evaluation_metrics,
    read_configuration_and_data_file,
    load_network_from_file,
)

dst = "breast_cancer"


train_file = f"{dst}/bcx.train.txt"
test_file = f"{dst}/bcx.test.txt"
(
    num_samples,
    num_features,
    n_outputs,
    train_dataset,
) = read_configuration_and_data_file(train_file)

test_dataset = read_configuration_and_data_file(test_file)[3]

print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")
print(f"Number of outputs: {n_outputs}")

NUM_EPOCHS = 200
# old_network = initialize_network(num_features + n_outputs - 1, 1, n_outputs)
old_network = initialize_network(num_features, 5, n_outputs)
print(old_network)
print()
network = load_network_from_file(f"{dst}/NNWDBC.txt")
print(network)

import json


def print_network_structure(network):
    # Create a copy of the network to avoid modifying the original network
    network_copy = {
        layer_name: [neuron.copy() for neuron in layer]
        for layer_name, layer in network.items()
    }

    # Replace the "weights" values with the length of the weights list
    for layer in network_copy.values():
        for neuron in layer:
            neuron["weights"] = len(neuron["weights"])

    # Print the JSON representation of the network
    print(json.dumps(network_copy, indent=4))


# Print the structure of the old_network
print("Old Network Structure:")
print_network_structure(old_network)

# Print the structure of the network
print("\nNew Network Structure:")
print_network_structure(network)


trained_network = train_network(network, train_dataset, 0.1, NUM_EPOCHS, n_outputs)

# TESTING ON UNSEEN DATA
test_predictions = []
train_predictions = []

# Make predictions for both the test and train datasets
for row in test_dataset:
    prediction = predict(trained_network, row)
    test_predictions.append(prediction)

for row in train_dataset:
    prediction = predict(trained_network, row)
    train_predictions.append(prediction)

# Compute evaluation metrics for the train and test datasets
train_evals = compute_evaluation_metrics(
    [row[-n_outputs:] for row in train_dataset], train_predictions
)
test_evals = compute_evaluation_metrics(
    [row[-n_outputs:] for row in test_dataset], test_predictions
)

# Print evaluations for the train and test datasets
print(f"Evaluations for the dataset on the train set are:{train_evals}")
print()
print(f"Evaluations for the dataset on the test set are:{test_evals}")

train_res = [train_dataset[i][-n_outputs:] for i in range(len(train_dataset))]
train_predictions = train_predictions

test_res = [test_dataset[i][-n_outputs:] for i in range(len(test_dataset))]
test_predictions = test_predictions

# number of incorrect predictions
train_incorrect = sum(
    [1 for i in range(len(train_predictions)) if train_predictions[i] != train_res[i]]
)
print(f"Number of incorrect predictions on the train set: {train_incorrect}")

test_incorrect = sum(
    [1 for i in range(len(test_predictions)) if test_predictions[i] != test_res[i]]
)

print(f"Number of incorrect predictions on the test set: {test_incorrect}")

print(trained_network)
