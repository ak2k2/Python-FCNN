import numpy as np
import random

# eq: dy / dx + py = q
# p: coefficient of y
# q: constant
# stability: 1 if stable, 0 if unstable


def generate_diff_eq_dataset(num_rows):
    dataset = []

    for _ in range(num_rows):
        # Randomly generate coefficients p and q for the diff eq: dy/dx + py = q
        p = random.uniform(-10, 10)
        q = random.uniform(-10, 10)

        # Determine stability: for simplicity, we'll consider the equation stable if p < 0
        stability = 1 if p < 0 else 0

        # Append to dataset
        dataset.append((p, q, stability))

    return dataset


# Generate a dataset with 500 rows
diff_eq_dataset = generate_diff_eq_dataset(500)


with open("deq.txt", "w") as file:
    for row in diff_eq_dataset:
        # Writing each row as space-separated values
        line = " ".join(map(str, row)) + "\n"
        file.write(line)
