# RT2. Exploring sample strategies

## Table of Contents
1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Function of module](#function-of-module)


## Requirements
- Python 3.9.13
- pandas
- numpy 1.23.5
- sklearn 1.0.2

## Usage

### Dataset
Credit score classification Dataset can be retreived from [HERE](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

- Filenames
    - Our original dataset `Master_data.csv`
    - Preprocessed dataset `Master_data_Preprocessed.csv`
- Contents
    - `Master_data.csv`, `Master_data_Preprocessed.csv` should be the csv files that stores the pandas dataframes that satisfy the following requirements:
        - The 'Binary_Credit_Score' column of the file is selected to be predicted by model
        
### Structure of code
The project have a structure as below:

```bash
├── RT2. Exploring sample strategies
│   ├── checkpoint
│   ├── data
│   │    ├── Master_data_Preprocessed.csv
│   │    ├── Master_data.csv
│   ├── figure
│   │    ├──err_fig.csv
│   │    ├── comb_hyper.csv
│   │    ├── acc_with_imbalance.csv
│   │    ├── average_acc_s.csv
│   ├── main.py
│   ├── Preprocess_data.py
│   ├── figure.py
│   ├── MLP.py
│   ├── cross_validation.py
│   ├── create_derived_data.py
│   ├── Sampling_strategy.py
```


### Running the main program
```bash
python main.py data/Master_data.csv
```
- You can also change the target path: python main.py \<DATA FILE>

### List of Arguments accepted
```--input_file``` String of the directory to the file containing master data. <br>

## Function of module
### `main.py`
- Entry point of the project.
- Handles the overall workflow including data preprocessing, splitting, model training and prediction
- Functions:
  - `main()`: Main function to run the entire process.

### `Preprocess_data.py`
- Contains functions for data preprocessing.
- Functions:
  - `preprocess_csv(input_file, output_file)`: Preprocesses the input CSV file and saves the output and return preprocessed data.
  - `split_Xy(data, column)`: Splits the data into features (X) and labels (y) and return X, y.

### `create_derived_data.py`
- Handles the creation of derived datasets.
  - `split_data(data, column, rate, num)`: Creates a derived data set by rate and based on the values of the specified column return derived data.

### `cross_validation.py`
- Contains functions for cross-validation and hyperparameter tuning.
- Functions:
  - `grid_search(X, y, param_grid, num_folds)`: Performs grid search for hyperparameter tuning.
  - `initializeMLP_with_bestHyper(params, X, y)`: Initializes the MLP model with the best hyperparameters.

### `Sampling_strategy.py`
- Contains functions for different sampling strategies.
- Functions:
  - `oversample()`: Implements oversampling strategy and return sampled data.
  - `undersample()`: Implements undersampling strategy and return sampled data.
  - `smote()`: Implements SMOTE strategy and return sampled data.
  - `check_imbalance(strategy, data, rates, model)`: Checks the imbalance in the dataset using the specified strategy.

### `MLP.py`
- Implements the Multi-Layer Perceptron (MLP) model.
- Module:
  - `Layer`: Define the layer structure of the neural network, including input, hidden, output layer.
  - `Optimizer`: Define the basic structure of the optimizer
  - `MLP`: Define the overall structure of the MLP, including inilitialization, forward, backward, fit, preidct.
- Functions:
  - `sigmod`: Defines a sigmoid function.
  - `show_error_plot()`: Displays the error in the training process as plot.
  - `compute_accuracy()`: Computes the accuracy of the model prediction.
  
### `figure.py`
- Contains functions for ploting figures
- Functions:
  - `plt_err_with_bestHyper(best_params, X, y)`: Plot error in the training process based on the best hyperprarmeter.
  - `plt_hyper_comb(data)`: Plot the accuracy of prediction with different combination of hyperparamter.
  - `plt_acc_imbalance(data)`: Plot the accuracy of prediction based on different sampling strategy and different imbalance degree.