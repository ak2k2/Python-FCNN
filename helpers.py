def accuracy_score(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        # Check if both the actual and predicted values are lists
        if isinstance(actual[i], list) and isinstance(predicted[i], list):
            if len(actual[i]) == len(predicted[i]) and all(
                a == p for a, p in zip(actual[i], predicted[i])
            ):
                correct += 1
        # Check if both are single-element lists or single elements
        elif actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0


def F1_score(actual, predicted):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(actual)):
        # Check if the predicted value is a list, and if so, take its first element
        predicted_val = (
            predicted[i][0] if isinstance(predicted[i], list) else predicted[i]
        )

        if actual[i] == predicted_val:
            TP += 1
        elif actual[i] != predicted_val:
            FP += 1
            FN += 1

    # Handle cases where TP, FP, or FN are zero to avoid ZeroDivisionError
    if TP == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return F1 * 100.0


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


# # Example usage:
# actual = [[1, 0], [0, 1], [1, 0], [1, 1], [0, 0]]
# predicted = [[1, 0], [0, 1], [0, 0], [1, 1], [0, 0]]
# metrics = compute_evaluation_metrics(actual, predicted)
# print(metrics)
