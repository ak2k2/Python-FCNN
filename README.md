## Running Testcases

1. To train a network on a given testcase (dataset, initial weights, and hyper-parameters).
```zsh
$ python3 src/train.py
# Enter the path for the network file: -> ini.txt
# Enter the path for the training file: -> tr.txt
# Enter the path for the output file: -> trained.txt 
# Enter the number of epochs for training: -> 200
# Enter the learning rate: -> 0.1
Succesfully trained network and saved to: 'trained.txt'
```

2. To preform validation and compute metrics on a test set.

```
Enter the name of the train dataset file (e.g., 'train_dataset.txt'): --> tr.txt
Enter the name of the test dataset file (e.g., 'test_dataset.txt'): 
-> te.txt
Enter the name of the initial architecture file (e.g., 'initial_weights.txt'): 
-> i.txt
```

## Custom Dataset Card

NOTE: Generate_[...].py is misnamed (generate_balanced_diff_eq_dataset). It was originally used to generate 2nd order ODE and preform binary classification on whether or not the solution converged.

Currently, the dataset contains 4 columns, 'a', 'b', 'c' which represent a simple quadratic equation of the form $$Y = ax^2 + bx + c $$
The discriminant is given by: $$b^2 + 4ac$$
1. **Prediction Target**: The classifier predicts whether the quadratic equation is "stable" or not, based on the criteria defined in the `check_stability` function. This is entirely arbitrary and was chosen for testing purposes. 
    
2. **Criteria for Stability**:
    - The script considers the equation "stable" if both roots are negative. This is derived from the sign of the roots calculated using the quadratic formula.
    - The script first checks the discriminant. If it's negative, the roots are complex, and the equation is automatically considered "unstable" (since complex roots can't be negative).
    - If the discriminant is non-negative, the script calculates the actual roots. The equation is "stable" only if both roots are negative.
    
3. **Generates Data Sets**: The `generate_balanced_diff_eq_dataset` function creates datasets for the coefficients of this quadratic equation, ensuring a balance between "stable" and "unstable" instances (as determined by the `check_stability` function).

The first 6 rows of the dataset look like this:
```
500 3 1
0.5026557363921625 0.05780456587088212 0.8151417184545711 0
0.11792892594714219 0.2800562641709723 0.24775920716193048 1
0.4155493342803154 0.009158494125059757 0.17505233696928446 1
0.5906299546606415 0.989591167946368 0.45895029119300856 0
0.3258498989350025 0.18635639808051324 0.0505978668893603 1
```

The `generate_initial_weights` function generates pseudo random weights and formats the file to match the expected NN architecture given the number of hidden nodes specified.

RNGs are seeded to deterministic values for the train and test dataset.

Ex. 
1. Generate a 500 row train set, and a 500 row test set.
2. Train for 2000 epochs using a learning rate of 0.1
3. Testing reveals the following results:
   
233 17 17 233 0.932 0.932 0.932 0.932
0.932 0.932 0.932 0.932
0.932 0.932 0.932 0.932