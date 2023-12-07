import math
import random


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


def forward_propagate(network, row):
    inputs = row
    for layer in network.values():
        new_inputs = []
        for neuron in layer:
            activation = neuron["weights"][-1]  # Bias
            for i in range(len(neuron["weights"]) - 1):
                activation += neuron["weights"][i] * inputs[i]
            neuron["output"] = sigmoid(activation)
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
    """
    network: FF_NN
    train: training data
    l_rate: learning rate
    n_epoch: number of epochs
    n_outputs: number of outputs
    """
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1  # ONE-HOT ENCODING the last column
            sum_error += cross_entropy(expected, outputs)
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, l_rate, sum_error))

    return network


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
