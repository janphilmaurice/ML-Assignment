import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from MLP import MLP, Adam

def grid_search(X, y, param_grid, num_folds=5):
    np.random.seed(42)
    results = []

    # Generate all combinations of hyperparameters
    #from itertools import product
    #keys, values = zip(*param_grid.items())
    #param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    param_combinations = list(ParameterGrid(param_grid))

    # Perform K-Fold Cross Validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for params in param_combinations:
        fold_accuracies = []
        print(f"Testing parameters: {params}")

        for train_idx, val_idx in kf.split(X):
            # Split the data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and train the MLP model
            optimizer_class = Adam 
            optimizer = optimizer_class()  # Initialize optimizer with default params
            mlp = MLP(
                NI=X.shape[1],
                NO=len(np.unique(y)),
                NHs=params["NHs"],
                lr=params["lr"],
                optimizer=optimizer,
            )

            mlp.fit(X_train, y_train, epoch=params["epoch"], batch_size=32)

            # Predict on validation data
            y_pred = mlp.predict(X_val)

            # Compute accuracy for classification tasks
            acc = accuracy_score(y_val, y_pred)
            fold_accuracies.append(acc)

        # Average accuracy across folds
        avg_acc = np.mean(fold_accuracies)
        print(f"Average Accuracy for parameters {params}: {avg_acc}")
        results.append((params, avg_acc))

    # Return best parameters and their performance
    best_params = max(results, key=lambda x: x[1])
    print(f"Best Parameters: {best_params[0]} with Accuracy: {best_params[1]}")
    return best_params, list(map(lambda x: x[1], results))

def initializeMLP_with_bestHyper(best_params, X, y):
    optimizer_class = Adam  # params["optimizer"]
    optimizer = optimizer_class()
    param_grid = {
        "NHs": [best_params['NHs']],  # Hidden layer configurations
        "lr": [best_params['lr']],  # Learning rates
        "batch_size": best_params['NHs'],  # Batch sizes
        "epoch": [200],  # Number of epochs
    }

    #keys, values = zip(*param_grid.items())
    # Generate all parameter combinations using ParameterGrid
    param_combinations = list(ParameterGrid(param_grid))

    for params in param_combinations:
        mlp = MLP(
            NI=X.shape[1],
            NO=len(np.unique(y)),
            NHs=params["NHs"],
            lr=params["lr"],
            optimizer=optimizer,
        )

    return mlp