import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from Preprocess_data import split_data, split_Xy
from cross_validation import initializeMLP_with_bestHyper


def oversample():
    print('Test oversampling!')
    Random_oversampling = RandomOverSampler(sampling_strategy="not majority")
    return Random_oversampling.fit_resample

def undersample():
    print('Test undersampling!')
    Random_Undersampling = RandomUnderSampler(sampling_strategy="not minority")
    return Random_Undersampling.fit_resample

def smote():
    print('Test SMOTE!')
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    return smote.fit_resample

def strategy_test(strategy, X, y, model):
    #X = data.loc[:, data.columns != 'Binary_Credit_Score'].values  # inputs
    #y = data['Binary_Credit_Score']
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