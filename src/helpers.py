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

        return network, Ni, Nh, No

    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def evaluate_predictions(true_labels, predictions):
    # Initialize the counters for micro-average
    micro_A = micro_B = micro_C = micro_D = 0
    class_metrics = []

    for class_index in range(len(true_labels[0])):
        A = B = C = D = 0
        # Count A, B, C, D for the current class
        for i in range(len(true_labels)):
            if (
                true_labels[i][class_index] == 1 and predictions[i][class_index] == 1
            ):  # predicted 1 and actual 1
                A += 1  # actual: 1, predicted: 1
            if true_labels[i][class_index] == 0 and predictions[i][class_index] == 1:
                B += 1  # actual: 0, predicted: 1
            if true_labels[i][class_index] == 1 and predictions[i][class_index] == 0:
                C += 1  # actual: 1, predicted: 0
            if true_labels[i][class_index] == 0 and predictions[i][class_index] == 0:
                D += 1  # actual: 0, predicted: 0

        # Update micro-average counts
        micro_A += A
        micro_B += B
        micro_C += C
        micro_D += D

        # Compute metrics for the current class
        accuracy = (A + D) / (A + B + C + D) if (A + B + C + D) > 0 else 0
        precision = A / (A + B) if (A + B) > 0 else 0
        recall = A / (A + C) if (A + C) > 0 else 0
        F1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Store the metrics for macro-average calculation
        class_metrics.append((A, B, C, D, accuracy, precision, recall, F1))

    # Compute micro-averaged metrics
    micro_accuracy = (micro_A + micro_D) / (micro_A + micro_B + micro_C + micro_D)
    micro_precision = micro_A / (micro_A + micro_B)
    micro_recall = micro_A / (micro_A + micro_C)
    micro_F1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    # Compute macro-averaged metrics
    macro_accuracy = sum(x[4] for x in class_metrics) / len(class_metrics)
    macro_precision = sum(x[5] for x in class_metrics) / len(class_metrics)
    macro_recall = sum(x[6] for x in class_metrics) / len(class_metrics)
    macro_F1 = sum(x[7] for x in class_metrics) / len(class_metrics)

    return (
        class_metrics,
        (micro_accuracy, micro_precision, micro_recall, micro_F1),
        (macro_accuracy, macro_precision, macro_recall, macro_F1),
    )
