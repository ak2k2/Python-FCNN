import math
import random

import numpy as np


def initialize_network(n_features, n_hidden, n_outputs):
    network = {
        "hidden_layer": [
            {"weights": [random.uniform(-1.0, 1.0) for i in range(n_features + 1)]}
            for j in range(n_hidden)
        ],
        "output_layer": [
            {"weights": [random.uniform(-1.0, 1.0) for i in range(n_hidden + 1)]}
            for j in range(n_outputs)
        ],
    }

    return network


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(output):
    return output * (1.0 - output)


def cross_entropy(actual, predicted):
    return -sum(
        [actual[i] * math.log(predicted[i] + 1e-15) for i in range(len(actual))]
    )


def forward_propagate(network, row):
    inputs = row
    for layer in network.values():
        new_inputs = []
        for neuron in layer:
            # Bias should be the first weight in the weights array and should be used with a fixed input of -1
            activation = (
                neuron["weights"][0] * -1
            )  # Using the first weight as bias with fixed input -1
            for i in range(1, len(neuron["weights"])):
                activation += (
                    neuron["weights"][i] * inputs[i - 1]
                )  # Adjust index to start from second weight
            neuron["output"] = sigmoid(activation)
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, expected):
    for i in reversed(list(network.keys())):
        layer = network[i]
        errors = list()
        if i == "output_layer":
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron["output"])
        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network["output_layer"]:
                    error += neuron["weights"][j] * neuron["delta"]
                errors.append(error)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron["delta"] = errors[j] * sigmoid_derivative(neuron["output"])


def update_weights(network, row, l_rate):
    for i, layer in enumerate(network.values()):
        inputs = [-1.0] + row[:-1]  # Bias input of -1 added at the beginning
        if (
            i != 0
        ):  # For layers beyond the first, the inputs are the outputs of the previous layer's neurons
            inputs = [-1.0] + [
                neuron["output"] for neuron in network[list(network.keys())[i - 1]]
            ]
        for neuron in layer:
            for j in range(len(inputs)):
                neuron["weights"][j] += l_rate * neuron["delta"] * inputs[j]


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [
                row[-n_outputs + i] for i in range(n_outputs)
            ]  # Extracting the one-hot encoded target
            sum_error += cross_entropy(expected, outputs)
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, l_rate, sum_error))

    return network


def predict(network, row, threshold=0.5):
    outputs = forward_propagate(network, row)
    return [int(output > threshold) for output in outputs]
