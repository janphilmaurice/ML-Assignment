# ML-Assignment

## Table of Contents
1. [Project Structure](#project-structure)
2. [Module Functionalities](#module-functionalities)
3. [Running Experiments](#running-experiments)
4. [Expected Behaviors](#expected-behaviors)
5. [Code Documentation](#code-documentation)

## Project Structure

## Module Functionalities

### `Main.py`
- Entry point of the project.
- Handles the overall workflow including data preprocessing, splitting, and model training.
- Functions:
  - `main()`: Main function to run the entire process.

### `Preprocess_data.py`
- Contains functions for data preprocessing.
- Functions:
  - `preprocess_csv(input_file, output_file)`: Preprocesses the input CSV file and saves the output.
  - `split_data(data, column, rate, num)`: Splits the data into two samples based on the specified column.
  - `split_Xy(data, column)`: Splits the data into features (X) and labels (y).

### `create_derived_data.py`
- Handles the creation of derived datasets.

### `cross_validation.py`
- Contains functions for cross-validation and hyperparameter tuning.
- Functions:
  - `grid_search(X, y, param_grid, num_folds)`: Performs grid search for hyperparameter tuning.
  - `initializeMLP_with_bestHyper(params, X, y)`: Initializes the MLP model with the best hyperparameters.

### `Sampling_strategy.py`
- Contains functions for different sampling strategies.
- Functions:
  - `oversample()`: Implements oversampling strategy.
  - `undersample()`: Implements undersampling strategy.
  - `smote()`: Implements SMOTE strategy.
  - `check_imbalance(strategy, data, rates, model)`: Checks the imbalance in the dataset using the specified strategy.

### `MLP.py`
- Implements the Multi-Layer Perceptron (MLP) model.
- Functions:
  - `write_train_data(filename)`: Writes training data to a file.
  - `show_error_plot()`: Displays the error plot.
  - `compute_accuracy()`: Computes the accuracy of the model.

## Running Experiments

1. **Preprocess Data**:
   python Coding/Main.py <DATA FILE>