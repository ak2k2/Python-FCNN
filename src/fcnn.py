import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def forward_propagate(network, inputs):
    inputs = [-1] + inputs  # Add the fixed input for bias
    hidden_inputs = []
    for neuron in network["hidden_layer"]:
        total_input = neuron["weights"][0] * -1
        for i in range(len(inputs) - 1):
            total_input += neuron["weights"][i + 1] * inputs[i + 1]
        neuron["output"] = sigmoid(total_input)
        hidden_inputs.append(neuron["output"])
    hidden_inputs = [-1] + hidden_inputs
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
    # Error: Output layer
    for i, neuron in enumerate(network["output_layer"]):
        error = expected[i] - neuron["output"]
        neuron["delta"] = error * sigmoid_derivative(neuron["output"])

    # Error: Hidden layer
    for i, neuron in enumerate(network["hidden_layer"]):
        error = sum(
            [
                network["output_layer"][j]["weights"][i + 1]
                * network["output_layer"][j]["delta"]
                for j in range(len(network["output_layer"]))
            ]
        )
        neuron["delta"] = error * sigmoid_derivative(neuron["output"])


def update_weights(network, row, l_rate, n_outputs):
    # Hidden layer weights
    inputs = [-1] + row[:-n_outputs]  # Add the fixed input for bias

    for i, neuron in enumerate(network["hidden_layer"]):
        for j in range(len(inputs)):
            neuron["weights"][j] += l_rate * neuron["delta"] * inputs[j]

    # Output layer weights
    inputs = [-1] + [
        neuron["output"] for neuron in network["hidden_layer"]
    ]  # Add fixed input for bias
    for i, neuron in enumerate(network["output_layer"]):
        for j in range(len(inputs)):
            neuron["weights"][j] += l_rate * neuron["delta"] * inputs[j]


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row[:-n_outputs])
            # The expected output is simply the last element of the row
            expected = row[-n_outputs:]
            # print(f"Debug - Expected: {expected}")
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate, n_outputs)
    return network


def predict(network, row, n_outputs):
    """
    Predicts the output for a given input row using the neural network.
    For binary classification (n_outputs=1), it returns a single integer 0 or 1.
    For multi-label classification (n_outputs>1), it returns a list of binary integers.

    Parameters:
    network: A trained neural network.
    row: A list representing an input row with features and labels.
    n_outputs: The number of outputs the network should predict.

    Returns:
    The prediction, which could be a binary value or a list of binary values.
    """
    # Exclude the labels (last 'n_outputs' element(s)) from the input features
    inputs = row[:-n_outputs]
    outputs = forward_propagate(network, inputs)

    # Binary classification case
    if n_outputs == 1:
        return 1 if outputs[0] > 0.5 else 0

    else:
        # Multi-label classification case
        return [1 if output > 0.5 else 0 for output in outputs]
