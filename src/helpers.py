def read_configuration_and_data_file(file_path: str):
    """
    Reads the configuration from the first line and data from the rest of the file.

    Parameters:
    file_path (str): The path to the file containing the configuration and data.

    Returns:
    tuple: A tuple containing the number of input nodes, number of hidden nodes,
           number of output nodes, and the contents of the file as a list of lists.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Extract the configuration from the first line
        config = lines[0].split()
        if len(config) != 3:
            raise ValueError("Configuration line must contain exactly three integers.")
        num_samples, num_features, n_outputs = map(int, config)

        # Extract the data from the remaining lines
        contents = []
        for line in lines[1:]:
            row = []
            items = line.split()

            # Process feature columns as floats
            for item in items[:num_features]:
                row.append(float(item))

            # Process output columns as integers
            for item in items[num_features:]:
                row.append(int(item))

            contents.append(row)

        return num_samples, num_features, n_outputs, contents
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None


def load_network_from_file(file_path: str):
    """
    Loads the neural network weights from a file.

    Parameters:
    file_path (str): The path to the file containing the neural network weights.

    Returns:
    dict: A dictionary representing the neural network with its layers and weights.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Extract the configuration from the first line
        config = lines[0].split()
        if len(config) != 3:
            raise ValueError("Configuration line must contain exactly three integers.")
        Ni, Nh, No = map(int, config)

        # Initialize the network structure
        network = {"hidden_layer": [], "output_layer": []}

        # Extract weights for hidden layer
        for i in range(1, 1 + Nh):
            weights = [float(w) for w in lines[i].split()]
            if len(weights) != Ni + 1:
                raise ValueError(
                    f"Expected {Ni + 1} weights, got {len(weights)} for a hidden node."
                )
            network["hidden_layer"].append({"weights": weights})

        # Extract weights for output layer
        for i in range(1 + Nh, 1 + Nh + No):
            weights = [float(w) for w in lines[i].split()]
            if len(weights) != Nh + 1:
                raise ValueError(
                    f"Expected {Nh + 1} weights, got {len(weights)} for an output node."
                )
            network["output_layer"].append({"weights": weights})

        return network

    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def compute_binary_classification_metrics(actual, predicted):
    """
    Computes evaluation metrics for binary classification.

    Parameters:
    actual (list): Actual classes.
    predicted (list): Predicted classes.

    Returns:
    metrics (dict): A dictionary with accuracy, precision, recall, and F1 score.
    """
    TP = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
    TN = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)
    FP = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
    FN = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)

    accuracy = (TP + TN) / len(actual)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1_score,
    }
