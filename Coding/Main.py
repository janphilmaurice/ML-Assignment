import os
import time

from Preprocess_data import preprocess_csv, split_Xy
from create_derived_data import split_data
from cross_validation import grid_search
from Sampling_strategy import oversample, undersample, smote, check_imbalance
from figure import plt_err_with_bestHyper, plt_hyper_comb, plt_acc_imbalance, plt_s_avg


def main():
    start_time = time.time()

    current_dir = os.getcwd()
    input_file = os.path.join(current_dir, "data/train.csv")  # change data path here
    output_file = os.path.join(current_dir, "data/train_Preprocessed.csv")
    print('Start preprocessing!\n')
    preprocessed_data = preprocess_csv(input_file, output_file)
    #print(preprocessed_data.isnull().any())

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

    best_params, hyper_accs = grid_search(balanced_X, balanced_y, param_grid, num_folds=5)
    plt_hyper_comb(hyper_accs)
    print('Finish grid search!\n')

    ## using strategy to train mlp
    # initialize model

    print('Draw the error plot of training with best hyperparameter\n')
    plt_err_with_bestHyper(best_params[0], balanced_X, balanced_y)

    print('Start using strategy to sample data!\n')
    # set the degree of imbalance
    rates = [(0.1,0.9),(0.2,0.8),(0.3,0.7),(0.4,0.6),(0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1)]
    stretegy = oversample()
    oversample_accs = check_imbalance(best_params[0], strategy=stretegy,
                    data=preprocessed_data, rates=rates)

    stretegy = undersample()
    undersample_accs = check_imbalance(best_params[0], strategy=stretegy,
                    data=preprocessed_data, rates=rates)

    stretegy = smote()
    smote_accs = check_imbalance(best_params[0], strategy=stretegy,
                    data=preprocessed_data, rates=rates)

    plt_s_avg([oversample_accs, undersample_accs, smote_accs])
    plt_acc_imbalance([oversample_accs, undersample_accs, smote_accs])



    end_time = time.time()

    # calculate time
    elapsed_time = end_time - start_time
    print(f"Program time:: {elapsed_time:.2f} s")

if __name__ == '__main__':
    main()




