def compute_evaluation_metrics(actual, predicted):
    """
    Computes evaluation metrics for binary classification per class.

    Parameters:
    actual (list of lists): Actual classes.
    predicted (list of lists): Predicted classes.

    Returns:
    metrics (list of dicts): A list of dictionaries with accuracy, precision, recall, and F1 score for each class.
    """

    # Initialize variables to hold the global counts for micro-averaging
    micro_TP = 0
    micro_FP = 0
    micro_FN = 0
    micro_TN = 0

    # List to hold metrics for each class
    metrics = []

    # Calculate metrics for each class
    for class_index in range(
        len(actual[0])
    ):  # Assume all rows have the same number of classes
        TP = FP = FN = TN = 0
        for i in range(len(actual)):
            if actual[i][class_index] == 1 and predicted[i][class_index] == 1:
                TP += 1
            elif actual[i][class_index] == 0 and predicted[i][class_index] == 1:
                FP += 1
            elif actual[i][class_index] == 1 and predicted[i][class_index] == 0:
                FN += 1
            elif actual[i][class_index] == 0 and predicted[i][class_index] == 0:
                TN += 1

        # Update micro-averaged counts
        micro_TP += TP
        micro_FP += FP
        micro_FN += FN
        micro_TN += TN

        # Calculate metrics for the current class
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0
            else 0
        )

        # Add the metrics for the current class to the list
        metrics.append(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "F1": f1_score,
            }
        )

    # Calculate micro-averaged metrics
    micro_accuracy = (micro_TP + micro_TN) / (micro_TP + micro_FP + micro_FN + micro_TN)
    micro_precision = (
        micro_TP / (micro_TP + micro_FP) if micro_TP + micro_FP != 0 else 0
    )
    micro_recall = micro_TP / (micro_TP + micro_FN) if micro_TP + micro_FN != 0 else 0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall != 0
        else 0
    )

    # Add micro-averaged metrics to the list
    metrics.append(
        {
            "micro_accuracy": micro_accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_F1": micro_f1,
        }
    )

    # Calculate macro-averaged metrics
    macro_accuracy = sum(metric["accuracy"] for metric in metrics[:-1]) / len(
        metrics[:-1]
    )
    macro_precision = sum(metric["precision"] for metric in metrics[:-1]) / len(
        metrics[:-1]
    )
    macro_recall = sum(metric["recall"] for metric in metrics[:-1]) / len(metrics[:-1])
    macro_f1 = sum(metric["F1"] for metric in metrics[:-1]) / len(metrics[:-1])

    # Add macro-averaged metrics to the list
    metrics.append(
        {
            "macro_accuracy": macro_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_F1": macro_f1,
        }
    )

    return metrics


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
