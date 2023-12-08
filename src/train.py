from helpers import *
from sable_nn import predict, train_network


def main(
    network_file_path: str,
    train_file_path: str,
    test_file_path: str,
    n_epochs: int = 100,
    lr: float = 0.1,
):
    initial_network = load_network_from_file(network_file_path)

    num_samples, num_features, n_outputs, train_data = read_configuration_and_data_file(
        train_file_path
    )

    _, _, _, test_data = read_configuration_and_data_file(test_file_path)

    trained_network = train_network(
        initial_network, train_data, lr, n_epochs, n_outputs
    )

    train_predictions = [predict(trained_network, row) for row in train_data]
    test_predictions = [predict(trained_network, row) for row in test_data]

    train_evals = compute_binary_classification_metrics(
        [x[-1] for x in train_data], train_predictions
    )

    test_evals = compute_binary_classification_metrics(
        [x[-1] for x in test_data], test_predictions
    )
    print("-" * 50)
    print("Train set evaluation:")
    print(f"Accuracy: {train_evals['accuracy']}")
    print(f"Precision: {train_evals['precision']}")
    print(f"Recall: {train_evals['recall']}")
    print(f"F1: {train_evals['F1']}")
    print("-" * 50)
    print("Test set evaluation:")
    print(f"Accuracy: {test_evals['accuracy']}")
    print(f"Precision: {test_evals['precision']}")
    print(f"Recall: {test_evals['recall']}")
    print(f"F1: {test_evals['F1']}")
    print("-" * 50)

    # return trained network but all weights are rounded to 3 decimal places
    return {
        "hidden_layer": [
            {
                "weights": [round(weight, 3) for weight in neuron["weights"]],
                "output": neuron["output"],
                "delta": neuron["delta"],
            }
            for neuron in trained_network["hidden_layer"]
        ],
        "output_layer": [
            {
                "weights": [round(weight, 3) for weight in neuron["weights"]],
                "output": neuron["output"],
                "delta": neuron["delta"],
            }
            for neuron in trained_network["output_layer"]
        ],
    }


if __name__ == "__main__":
    # tn = main(
    #     network_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/NNWDBC.txt",
    #     train_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/WDBC_train.txt",
    #     test_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/WDBC_test.txt",
    #     n_epochs=100,
    #     lr=0.1,
    # )
    tn = main(
        network_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/NNGRADES.txt",
        train_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/GRADES_train.txt",
        test_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/GRADES_test.txt",
        n_epochs=100,
        lr=0.1,
    )
    print(tn)
