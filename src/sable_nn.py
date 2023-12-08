import random
import math


# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Initialize a network with the given structure
def initialize_network(
    n_inputs, n_hidden, n_outputs, hidden_layer_weights, output_layer_weights
):
    network = {
        "hidden_layer": [{"weights": weights} for weights in hidden_layer_weights],
        "output_layer": [{"weights": weights} for weights in output_layer_weights],
    }
    return network


# Forward propagate input to a network output
def forward_propagate(network, inputs):
    inputs = [-1] + inputs  # Add the fixed input for bias
    hidden_inputs = []
    for neuron in network["hidden_layer"]:
        # Initialize weighted sum with the bias weight times fixed input
        total_input = neuron["weights"][0] * -1
        # Add the rest of the weighted inputs
        for i in range(len(inputs) - 1):
            total_input += neuron["weights"][i + 1] * inputs[i + 1]
        neuron["output"] = sigmoid(total_input)
        hidden_inputs.append(neuron["output"])
    hidden_inputs = [
        -1
    ] + hidden_inputs  # Add the fixed input for bias in the output layer
    output_inputs = []
    for neuron in network["output_layer"]:
        # Initialize weighted sum with the bias weight times fixed input
        total_input = neuron["weights"][0] * -1
        # Add the rest of the weighted inputs
        for i in range(len(hidden_inputs) - 1):
            total_input += neuron["weights"][i + 1] * hidden_inputs[i + 1]
        neuron["output"] = sigmoid(total_input)
        output_inputs.append(neuron["output"])
    return output_inputs


# Backward propagate error and store in neurons
def backward_propagate_error(network, expected):
    # Calculate error for output layer
    for i, neuron in enumerate(network["output_layer"]):
        error = expected[i] - neuron["output"]
        neuron["delta"] = error * sigmoid_derivative(neuron["output"])

    # Calculate error for hidden layer
    for i, neuron in enumerate(network["hidden_layer"]):
        error = sum(
            [
                network["output_layer"][j]["weights"][i + 1]
                * network["output_layer"][j]["delta"]
                for j in range(len(network["output_layer"]))
            ]
        )
        neuron["delta"] = error * sigmoid_derivative(neuron["output"])


# Update network weights with error
def update_weights(network, row, l_rate):
    # Update weights for the hidden layer
    inputs = [-1] + row[:-1]  # Add the fixed input for bias
    for i, neuron in enumerate(network["hidden_layer"]):
        for j in range(len(inputs)):
            neuron["weights"][j] += l_rate * neuron["delta"] * inputs[j]

    # Update weights for the output layer
    inputs = [-1] + [
        neuron["output"] for neuron in network["hidden_layer"]
    ]  # Add the fixed input for bias
    for i, neuron in enumerate(network["output_layer"]):
        for j in range(len(inputs)):
            neuron["weights"][j] += l_rate * neuron["delta"] * inputs[j]


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row[:-1])
            expected = [
                row[-1]
            ]  # The expected output is simply the last element of the row
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
    return network


def predict(network, row):
    # Exclude the label (last element) from the input features
    inputs = row[:-1]
    outputs = forward_propagate(network, inputs)
    # Use 0.5 as a threshold to decide the class
    return 1 if outputs[0] > 0.5 else 0


# # Example usage:
# n_inputs = # Number of features
# n_hidden = # Number of hidden neurons
# n_outputs = # Number of output neurons
# hidden_layer_weights = # Hidden layer weights as provided
# output_layer_weights = # Output layer weights as provided
# network = initialize_network(n_inputs, n_hidden, n_outputs, hidden_layer_weights, output_layer_weights)
# train = # Training dataset

# trained_network = train_network(network, train, 0.1, NUM_EPOCHS, n_outputs)
