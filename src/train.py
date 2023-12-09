from helpers import load_network_from_file, read_configuration_and_data_file
from sable_nn import train_network


def train_pipline(
    network_file_path: str,
    train_file_path: str,
    output_file_path: str,
    n_epochs: int = 100,
    lr: float = 0.1,
):
    initial_network, Ni, Nh, No = load_network_from_file(network_file_path)

    num_samples, num_features, n_outputs, train_data = read_configuration_and_data_file(
        train_file_path
    )

    trained_network = train_network(
        initial_network, train_data, lr, n_epochs, n_outputs
    )

    # write the trained network to a file
    with open(output_file_path, "w") as file:
        # first line contains Ni, Nh, No
        file.write(f"{Ni} {Nh} {No}\n")
        # next Nh lines contain the weights of edges pointing from input nodes to hidden nodes
        for neuron in trained_network["hidden_layer"]:
            rounded_weights = [round(weight, 3) for weight in neuron["weights"]]
            file.write(" ".join(map(str, rounded_weights)))
            file.write("\n")
        # next No lines contain the weights of edges pointing from hidden nodes to output nodes
        for neuron in trained_network["output_layer"]:
            rounded_weights = [round(weight, 3) for weight in neuron["weights"]]
            file.write(" ".join(map(str, rounded_weights)))
            file.write("\n")

    print(f"Trained network {trained_network}")


if __name__ == "__main__":
    # network_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/NNGRADES.txt"
    # train_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/GRADES_train.txt"
    # output_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/KAPOOR_TRAINED_GRADES.txt"

    network_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/NNWDBC.txt"
    train_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/WDBC_train.txt"
    output_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/KAPOOR_TRAINED_NNWDBC.txt"

    n_epochs = 100
    lr = 0.1

    trained_network = train_pipline(
        network_file_path=network_file_path,
        train_file_path=train_file_path,
        output_file_path=output_file_path,
        n_epochs=n_epochs,
        lr=lr,
    )
