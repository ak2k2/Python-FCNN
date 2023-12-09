from helpers import *
from sable_nn import predict, train_network


def main(
    network_file_path: str,
    train_file_path: str,
    test_file_path: str,
    n_epochs: int = 100,
    lr: float = 0.1,
):
    initial_network, _, _, _ = load_network_from_file(network_file_path)

    num_samples, num_features, n_outputs, train_data = read_configuration_and_data_file(
        train_file_path
    )

    _, _, _, test_data = read_configuration_and_data_file(test_file_path)

    trained_network = train_network(
        initial_network, train_data, lr, n_epochs, n_outputs
    )

    train_predictions = [predict(trained_network, row, n_outputs) for row in train_data]
    test_predictions = [predict(trained_network, row, n_outputs) for row in test_data]

    train_evals = compute_multiclass_classification_metrics(
        [x[-1] for x in train_data], train_predictions
    )

    test_evals = compute_multiclass_classification_metrics(
        [x[-1] for x in test_data], test_predictions
    )
    print("-" * 50)
    print("Training set evaluation:")
    print(train_evals)
    print("-" * 50)
    print("Test set evaluation:")
    print(test_evals)
    print("-" * 50)

    # return trained network but all weights are rounded to 3 decimal places
    return {
        "hidden_layer": [
            {
                "weights": [round(weight, 3) for weight in neuron["weights"]],
                "output": round(neuron["output"], 3),
                "delta": round(neuron["delta"], 3),
            }
            for neuron in trained_network["hidden_layer"]
        ],
        "output_layer": [
            {
                "weights": [round(weight, 3) for weight in neuron["weights"]],
                "output": round(neuron["output"], 3),
                "delta": round(neuron["delta"], 3),
            }
            for neuron in trained_network["output_layer"]
        ],
    }


if __name__ == "__main__":
    tn = main(
        network_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/NNWDBC.txt",
        train_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/WDBC_train.txt",
        test_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/WDBC_test.txt",
        n_epochs=100,
        lr=0.1,
    )

    # # unit test WDBC
    print(
        tn
        == {
            "hidden_layer": [
                {
                    "weights": [
                        0.209,
                        0.262,
                        0.577,
                        0.788,
                        0.065,
                        0.938,
                        0.028,
                        0.451,
                        0.671,
                        0.876,
                        0.776,
                        0.141,
                        0.554,
                        0.98,
                        0.975,
                        0.506,
                        0.45,
                        0.404,
                        0.853,
                        0.571,
                        0.224,
                        0.901,
                        0.978,
                        0.298,
                        0.448,
                        0.778,
                        0.943,
                        0.398,
                        0.445,
                        0.343,
                        0.71,
                    ],
                    "output": 1.0,
                    "delta": 0.0,
                },
                {
                    "weights": [
                        0.216,
                        0.689,
                        0.983,
                        1.1,
                        0.94,
                        0.557,
                        0.272,
                        0.73,
                        0.579,
                        0.935,
                        0.412,
                        0.509,
                        0.787,
                        0.611,
                        0.212,
                        0.709,
                        0.745,
                        1.009,
                        0.264,
                        0.55,
                        0.577,
                        0.878,
                        0.719,
                        1.08,
                        0.399,
                        0.285,
                        1.036,
                        0.601,
                        0.675,
                        0.242,
                        0.764,
                    ],
                    "output": 1.0,
                    "delta": 0.0,
                },
                {
                    "weights": [
                        0.358,
                        0.897,
                        0.65,
                        0.471,
                        0.083,
                        0.845,
                        0.661,
                        0.223,
                        0.478,
                        0.556,
                        0.972,
                        0.15,
                        0.952,
                        0.089,
                        0.906,
                        0.749,
                        0.136,
                        0.643,
                        0.888,
                        1.043,
                        0.361,
                        0.976,
                        0.543,
                        0.822,
                        0.662,
                        0.774,
                        0.302,
                        0.551,
                        0.805,
                        0.912,
                        0.391,
                    ],
                    "output": 1.0,
                    "delta": 0.0,
                },
                {
                    "weights": [
                        0.759,
                        0.929,
                        0.876,
                        0.642,
                        0.537,
                        0.606,
                        0.938,
                        0.446,
                        0.918,
                        0.787,
                        0.922,
                        0.345,
                        0.213,
                        0.39,
                        0.219,
                        0.292,
                        0.91,
                        0.168,
                        0.125,
                        0.829,
                        0.979,
                        0.16,
                        0.242,
                        0.793,
                        0.68,
                        0.824,
                        0.833,
                        0.992,
                        0.901,
                        0.527,
                        0.103,
                    ],
                    "output": 1.0,
                    "delta": 0.0,
                },
                {
                    "weights": [
                        3.869,
                        0.55,
                        0.948,
                        0.43,
                        1.532,
                        -0.41,
                        -0.061,
                        0.998,
                        2.455,
                        -1.304,
                        -2.347,
                        1.122,
                        -0.121,
                        0.791,
                        0.818,
                        0.276,
                        -0.772,
                        -0.136,
                        -0.663,
                        -0.292,
                        -0.298,
                        1.122,
                        1.706,
                        1.365,
                        1.847,
                        0.149,
                        0.82,
                        1.317,
                        1.655,
                        0.576,
                        -0.193,
                    ],
                    "output": 0.998,
                    "delta": -0.0,
                },
            ],
            "output_layer": [
                {
                    "weights": [-0.226, 0.909, 1.244, 1.325, 0.628, -7.663],
                    "output": 0.035,
                    "delta": 0.0,
                }
            ],
        }
    )
    print("\n\n")
    tn2 = main(
        network_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/NNGRADES.txt",
        train_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/GRADES_train.txt",
        test_file_path="/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/GRADES_test.txt",
        n_epochs=100,
        lr=0.05,
    )

    print(tn2)
