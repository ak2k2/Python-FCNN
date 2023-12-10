import random

import numpy as np


# Function to check the stability of a differential equation
def check_stability(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return 0
    root1 = (-b + np.sqrt(discriminant)) / (2 * a)
    root2 = (-b - np.sqrt(discriminant)) / (2 * a)
    return 1 if root1 < 0 and root2 < 0 else 0


# Function to normalize a value
def min_max_normalize(value, min_val, max_val, new_min=0, new_max=1):
    return (value - min_val) / (max_val - min_val) * (new_max - new_min) + new_min


# Function to generate a balanced dataset for a differential equation
def generate_balanced_diff_eq_dataset(num_rows, a_range, b_range, c_range, seed):
    random.seed(seed)
    np.random.seed(seed)
    dataset = []
    stability_count = [0, 0]
    while len(dataset) < num_rows:
        a, b, c = (
            random.uniform(*a_range),
            random.uniform(*b_range),
            random.uniform(*c_range),
        )
        a_normalized, b_normalized, c_normalized = (
            min_max_normalize(a, *a_range),
            min_max_normalize(b, *b_range),
            min_max_normalize(c, *c_range),
        )
        stability = check_stability(a, b, c)
        if stability_count[stability] < num_rows / 2:
            stability_count[stability] += 1
            dataset.append((a_normalized, b_normalized, c_normalized, stability))

    random.shuffle(dataset)  # Shuffle the dataset before returning
    return dataset


# Function to generate initial weights for the neural network
def generate_initial_weights(num_input_features, num_hidden_layers, num_output_classes):
    weights = []
    for _ in range(num_hidden_layers):
        layer_weights = [
            round(random.uniform(-1, 1), 3) for _ in range(num_input_features + 1)
        ]
        weights.append(layer_weights)
    output_weights = [
        round(random.uniform(-1, 1), 3) for _ in range(num_hidden_layers + 1)
    ]
    weights.append(output_weights)
    return weights


# Main function
def main():
    # Input file names
    train_dataset_file_name = input(
        "Enter the name of the train dataset file (e.g., 'train_dataset.txt'): "
    )
    test_dataset_file_name = input(
        "Enter the name of the test dataset file (e.g., 'test_dataset.txt'): "
    )
    weights_file_name = input(
        "Enter the name of the initial architecture file (e.g., 'initial_weights.txt'): "
    )

    # Constants
    num_rows, num_input_features, num_output_classes = 500, 3, 1
    a_range, b_range, c_range = (-10, 10), (-15, 5), (-5, 15)

    # Train dataset generation
    train_dataset = generate_balanced_diff_eq_dataset(
        num_rows, a_range, b_range, c_range, 420
    )
    with open(train_dataset_file_name, "w") as file:
        file.write(f"{num_rows} {num_input_features} {num_output_classes}\n")
        for row in train_dataset:
            file.write(" ".join(map(str, row)) + "\n")
    print(f"Train dataset generated and saved to {train_dataset_file_name}")

    # Test dataset generation
    test_dataset = generate_balanced_diff_eq_dataset(
        num_rows, a_range, b_range, c_range, 69
    )
    with open(test_dataset_file_name, "w") as file:
        file.write(f"{num_rows} {num_input_features} {num_output_classes}\n")
        for row in test_dataset:
            file.write(" ".join(map(str, row)) + "\n")
    print(f"Test dataset generated and saved to {test_dataset_file_name}")

    # Initial weights generation
    num_hidden_layers = int(input("Enter the number of hidden layers: "))
    initial_weights = generate_initial_weights(
        num_input_features, num_hidden_layers, num_output_classes
    )
    with open(weights_file_name, "w") as file:
        file.write(f"{num_input_features} {num_hidden_layers} {num_output_classes}\n")
        for layer_weights in initial_weights:
            file.write(" ".join(map(str, layer_weights)) + "\n")
    print(f"Initial weights generated and saved to {weights_file_name}")


if __name__ == "__main__":
    main()
