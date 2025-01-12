import os
import time

from Preprocess_data import preprocess_csv, split_Xy
from create_derived_data import split_data
from cross_validation import grid_search, initializeMLP_with_bestHyper
from Sampling_strategy import oversample, undersample, smote, check_imbalance


def main():
    start_time = time.time()

    current_dir = os.getcwd()
    input_file = os.path.join(current_dir, "data/Master_Data.csv")  # change data path here
    output_file = os.path.join(current_dir, "data/Master_Data_Preprocessed.csv")
    print('Start preprocessing!\n')
    preprocessed_data = preprocess_csv(input_file, output_file)
    #print(preprocessed_data.isnull().any())
    #print(preprocessed_data['Binary_Credit_Score'])

    column='Binary_Credit_Score'
    balanced_data = split_data(preprocessed_data,column,rate=(0.5,0.5),num=5000)
    #print(balanced_data.isnull().any())

    # get balanced version
    balanced_X, balanced_y = split_Xy(balanced_data, column)

    # set hyperparam we want to test
    print('Start grid search!\n')
    param_grid = {
        "NHs": [[32], [64, 32], [128, 64, 32]],  # Hidden layer configurations
        "lr": [0.01, 0.001],  # Learning rates
        "epoch": [50,200],  # Number of epochs
    }

    # Perform grid search

    best_params = grid_search(balanced_X, balanced_y, param_grid, num_folds=5)
    print('Finish grid search!\n')

    ## using strategy to train mlp
    # initialize model
    mlp = initializeMLP_with_bestHyper(best_params[0], balanced_X, balanced_y)

    # set the degree of imbalance
    print('Start using strategy to sample data!\n')
    rates = [(0.3,0.7),(0.2,0.8),(0.5,0.5),(0.8,0.2),(0.7,0.3)]
    stretegy = oversample()
    check_imbalance(strategy=stretegy,
                    data=preprocessed_data, rates=rates, model=mlp)

    stretegy = undersample()
    check_imbalance(strategy=stretegy,
                    data=preprocessed_data, rates=rates, model=mlp)

    stretegy = smote()
    check_imbalance(strategy=stretegy,
                    data=preprocessed_data, rates=rates, model=mlp)

    end_time = time.time()

    # calculate time
    elapsed_time = end_time - start_time
    print(f"Program time:: {elapsed_time:.2f} s")

if __name__ == '__main__':
    main()




