import math
import random
import numpy as np


def read_configuration_file(cfg_file: str) -> tuple[int, int, int]:
    """
    Reads the configuration file and returns a tuple of integers.
    """
    try:
        with open(
            cfg_file, "r"
        ) as f:  # read only the first line to find 3 space separated integers
            line = f.readline()
            num_input_nodes, num_hidden_nodes, num_output_nodes = map(int, line.split())

        print("#Input Nodes: ", num_input_nodes)
        print("#Number Hidden Nodes: ", num_hidden_nodes)
        print("#Number Output Nodes: ", num_output_nodes)

        return num_input_nodes, num_hidden_nodes, num_output_nodes

    except FileNotFoundError:
        print("Configuration file not found. Please check the path and try again.")


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = {
        "hidden_layer": [
            {"weights": [random.uniform(-1.0, 1.0) for i in range(n_inputs + 1)]}
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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def forward_propagate(network, row, n_outputs):
    inputs = row
    for layer_name, layer in network.items():
        new_inputs = []
        for neuron in layer:
            activation = neuron["weights"][-1]  # Bias
            for i in range(len(neuron["weights"]) - 1):
                activation += neuron["weights"][i] * inputs[i]
            neuron["output"] = (
                sigmoid(activation)
                if layer_name != "output_layer"
                else (softmax(activation) if n_outputs > 1 else sigmoid(activation))
            )
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    return inputs


def cross_entropy(actual, predicted):
    return -sum(
        [actual[i] * math.log(predicted[i] + 1e-15) for i in range(len(actual))]
    )


def sigmoid_derivative(output):
    return output * (1.0 - output)


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
        inputs = row[:-1]
        if i != 0:
            inputs = [
                neuron["output"] for neuron in network[list(network.keys())[i - 1]]
            ]
        for neuron in layer:
            for j in range(len(inputs)):
                neuron["weights"][j] += l_rate * neuron["delta"] * inputs[j]
            neuron["weights"][-1] += l_rate * neuron["delta"]  # Update bias


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1  # Adjusted for N classes
            sum_error += cross_entropy(expected, outputs)
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, l_rate, sum_error))

    return network


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def predict_multiclass(network, row, threshold=0.5):
    outputs = forward_propagate(network, row)
    # Apply threshold to each output
    prediction = [1 if output > threshold else 0 for output in outputs]
    # Enforce logical consistency
    return predict


def accuracy_score(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0


def F1_score(actual, predicted):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            TP += 1
        elif actual[i] != predicted[i]:
            FP += 1
            FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return F1
