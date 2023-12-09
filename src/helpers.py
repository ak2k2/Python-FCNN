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


def compute_multiclass_classification_metrics(actual, predicted):
    """
    Computes evaluation metrics for binary and multi-class classification, including micro and macro averages.

    Parameters:
    actual (list or list of list): Actual classes.
    predicted (list or list of list): Predicted classes.

    Returns:
    metrics (dict): A dictionary with per-class accuracy, precision, recall, and F1 score, and micro and macro-averaged metrics.
    """
    # If actual and predicted are not lists of lists, convert them to lists of lists (one-hot encoding)
    if isinstance(actual[0], int):
        actual = [[1 if i == a else 0 for i in range(max(actual) + 1)] for a in actual]
        predicted = [
            [1 if i == p else 0 for i in range(max(predicted) + 1)] for p in predicted
        ]

    n_classes = len(actual[0])
    class_metrics = []
    micro_TP = micro_FP = micro_FN = micro_TN = 0
    macro_accuracy = macro_precision = macro_recall = macro_f1 = 0

    # Calculate metrics per class
    for class_index in range(n_classes):
        TP = sum(
            1
            for a, p in zip(actual, predicted)
            if a[class_index] == 1 and p[class_index] == 1
        )
        TN = sum(
            1
            for a, p in zip(actual, predicted)
            if a[class_index] == 0 and p[class_index] == 0
        )
        FP = sum(
            1
            for a, p in zip(actual, predicted)
            if a[class_index] == 0 and p[class_index] == 1
        )
        FN = sum(
            1
            for a, p in zip(actual, predicted)
            if a[class_index] == 1 and p[class_index] == 0
        )

        # Update micro metrics
        micro_TP += TP
        micro_FP += FP
        micro_FN += FN
        micro_TN += TN

        accuracy = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN != 0 else 0
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0
            else 0
        )

        # Update macro metrics
        macro_accuracy += accuracy
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1_score

        class_metrics.append(
            {
                "A": TP,
                "B": FP,
                "C": FN,
                "D": TN,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "F1": f1_score,
            }
        )

    # Compute micro-averaged metrics
    micro_accuracy = (micro_TP + micro_TN) / (micro_TP + micro_FP + micro_FN + micro_TN)
    micro_precision = (
        micro_TP / (micro_TP + micro_FP) if micro_TP + micro_FP != 0 else 0
    )
    micro_recall = micro_TP / (micro_FN + micro_TP) if micro_TP + micro_FN != 0 else 0
    micro_f1_score = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall != 0
        else 0
    )

    # Compute macro-averaged metrics
    macro_accuracy /= n_classes
    macro_precision /= n_classes
    macro_recall /= n_classes
    macro_f1 /= n_classes

    # Combine class metrics with micro-averaged and macro-averaged metrics
    metrics = {
        "per_class": class_metrics,
        "micro_averaged": {
            "accuracy": micro_accuracy,
            "precision": micro_precision,
            "recall": micro_recall,
            "F1": micro_f1_score,
        },
        "macro_averaged": {
            "accuracy": macro_accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "F1": macro_f1,
        },
    }

    return metrics
