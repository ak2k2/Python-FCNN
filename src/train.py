import numpy as np
import pandas as pd

from fcnn import initialize_network, predict, train_network
from helpers import compute_evaluation_metrics, read_configuration_and_data_file

dataset_file = "bcx.train.txt"

(
    num_samples,
    num_features,
    n_outputs,
    train_dataset,
) = read_configuration_and_data_file("bcx.train.txt")

test_dataset = read_configuration_and_data_file("bcx.test.txt")[3]

print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")
print(f"Number of outputs: {n_outputs}")
print(train_dataset)

NUM_EPOCHS = 100
network = initialize_network(num_features, 1, n_outputs)
trained_network = train_network(network, train_dataset, 0.1, NUM_EPOCHS, n_outputs)

print(f"All of the hyperparameters of this NN are: {trained_network}")

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
