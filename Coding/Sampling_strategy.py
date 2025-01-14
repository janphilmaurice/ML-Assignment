import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors

from Preprocess_data import split_data, split_Xy
from cross_validation import initializeMLP_with_bestHyper


def oversample(X, y):
    # Find the number of samples for each class
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    # Initialize storage of new samples
    X_resampled = X.copy()
    y_resampled = y.copy()

    # Oversampling each class
    for class_label, count in zip(unique, counts):
        if count < max_count:
            # Extract samples belonging to the current category
            X_class = X[y == class_label]
            y_class = y[y == class_label]

            # Calculate the number of samples that need to be supplemented
            n_samples_to_add = max_count - count

            # Use random sampling to supplement the sample
            X_upsampled, y_upsampled = resample(X_class, y_class, 
                                                replace=True, 
                                                n_samples=n_samples_to_add, 
                                                random_state=42)

            # Append to a new dataset
            X_resampled = np.vstack((X_resampled, X_upsampled))
            y_resampled = np.hstack((y_resampled, y_upsampled))

    return X_resampled, y_resampled

def undersample(X, y):
    # Find the number of samples for each class
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    # Initialize storage of new samples
    X_resampled = []
    y_resampled = []

    # Downsample each class
    for class_label, count in zip(unique, counts):
        # Extract samples belonging to the current category
        X_class = X[y == class_label]
        y_class = y[y == class_label]

        # Use random downsampling
        X_downsampled, y_downsampled = resample(X_class, y_class, 
                                                replace=False, 
                                                n_samples=min_count, 
                                                random_state=42)

        # Append to a new dataset
        X_resampled.append(X_downsampled)
        y_resampled.append(y_downsampled)

    return np.vstack(X_resampled), np.hstack(y_resampled)



def smote(X, y, k_neighbors=5):
    # Find the number of samples for each class
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    # Initialize storage of new samples
    X_resampled = X.copy()
    y_resampled = y.copy()

    for class_label, count in zip(unique, counts):
        if count < max_count:
            # Extract samples belonging to the current category
            X_class = X[y == class_label]

            # Calculate the number of samples that need to be supplemented
            n_samples_to_add = max_count - count

            # Find k nearest neighbors
            nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_class)
            neighbors = nn.kneighbors(X_class, return_distance=False)

            # Generating synthetic samples
            synthetic_samples = []
            for _ in range(n_samples_to_add):
                # Randomly select a sample and its neighbors
                i = np.random.randint(len(X_class))
                neighbor_idx = neighbors[i, np.random.randint(1, k_neighbors + 1)]
                
                # Linear interpolation generates new samples
                diff = X_class[neighbor_idx] - X_class[i]
                synthetic_sample = X_class[i] + np.random.rand() * diff
                synthetic_samples.append(synthetic_sample)

            # Append to a new dataset
            X_resampled = np.vstack((X_resampled, synthetic_samples))
            y_resampled = np.hstack((y_resampled, np.full(len(synthetic_samples), class_label)))

    return X_resampled, y_resampled

def strategy_test(strategy, X, y, model):
    resample_X, resample_y = strategy(X, y)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        resample_X, resample_y, test_size=0.2, random_state=42, stratify=resample_y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

    model.fit(X_train, y_train, epoch=200, batch_size=32)
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    #print(f'val_accuracy:{val_accuracy}, test_accuracy: {test_accuracy}')

    if np.abs(val_accuracy - test_accuracy) > 0.1:
        return print('Overfitting!')

    #print(f'The result in this strategy in {rate} is: {test_accuracy*100:.2f}%')
    return test_accuracy

def check_imbalance(best_params, strategy=None, data=None, rates=None):
    #resample_X, resample_y = strategy(X, y)
    #num = max(resample_y[resample_y==0].shape[0],resample_y[resample_y==1].shape[0])
    accs=[]
    column = 'Binary_Credit_Score'
    for rate in rates:
        sample_data = split_data(data,column=column, rate=rate, num=5000)  # Divide the preprocessed data based on a given ratio
        X, y = split_Xy(sample_data, column)
        mlp = initializeMLP_with_bestHyper(best_params, X, y)
        #X = sample_data.loc[:, sample_data.columns != 'Binary_Credit_Score'].values  # inputs
        #y = sample_data['Binary_Credit_Score']
        acc = strategy_test(strategy, X, y, mlp)  # using strategy to sample
        # #print(f'rate: {rate}; acc: {acc}')
        accs.append(acc)
    for i in range(len(rates)):
        print(f'The accuracy based on rate{rates[i]} is {accs[i]*100:.2f}%')

    return accs