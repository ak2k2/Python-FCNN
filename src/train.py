from fcnn import train_network
from helpers import load_network_from_file, read_configuration_and_data_file


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


if __name__ == "__main__":
    network_file_path = input("Enter the path for the network file: ")
    train_file_path = input("Enter the path for the training file: ")
    output_file_path = input("Enter the path for the output file: ")

    n_epochs = int(input("Enter the number of epochs for training: "))
    lr = float(input("Enter the learning rate: "))

    trained_network = train_pipline(
        network_file_path=network_file_path,
        train_file_path=train_file_path,
        output_file_path=output_file_path,
        n_epochs=n_epochs,
        lr=lr,
    )

    print(f"Succesfully trained network and saved to: '{output_file_path}'")
