import numpy as np

from fomlads.plot.evaluations import plot_roc

from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from fomlads.model.classification import logistic_regression_prediction_probs

import pandas as pd
import matplotlib.pyplot as plt


def preprocess_csv(input_file, output_file):
    # Load the data from the input CSV file
    df = pd.read_csv(input_file)

    # Create a 'Customer_ID_Number'
    unique_customer_mapping = {customer_id: idx + 1 for idx, customer_id in enumerate(df['Customer_ID'].unique())}
    df['Customer_ID_Number'] = df['Customer_ID'].map(unique_customer_mapping)

    # Remove rows where 'Credit_Mix' & 'Changed_Credit_Limit' have the value "-"
    df = df[df['Credit_Mix'] != "_"]
    df = df[df['Changed_Credit_Limit'] != "_"]

    # Create a new column 'Binary_credit_mix' and 'Credit_History_Age_by_year'
    df['Binary_credit_mix'] = df['Credit_Mix'].apply(lambda x: "1" if x == "Good" else "0")
    df['Credit_History_Age_by_year'] = df['Credit_History_Age'].str.extract(r'(\d+)', expand=False)

    # Selects only apropiate values and eliminates "_"
    eliminate_hyphen = ['Age', 'Annual_Income', 'Outstanding_Debt',
                        'Num_of_Delayed_Payment', 'Num_of_Loan']
    for i in eliminate_hyphen:
        df[i] = df[i].astype(str).str.replace("_", "", regex=False).astype(float)

    df = df[(df['Num_Bank_Accounts'] >= 0) & (df['Num_Bank_Accounts'] <= 100)]
    df = df[(df['Num_Credit_Card'] >= 0) & (df['Num_Credit_Card'] <= 500)]
    df = df[(df['Num_of_Loan'] >= 0) & (df['Num_of_Loan'] <= 1000)]

    # Fills empty values with 0
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].fillna(0)
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].fillna(0)

    # Replace values in 'Month'
    df['Month'] = df['Month'].map({
        "September": 1,
        "October": 2,
        "November": 3,
        "December": 4
    })

    # Replace values in 'Payment_of_Min_Amount'
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({
        "NM": 0,
        "No": 1,
        "Yes": 2
    })

    # Rounds values to 2 decimals
    columns_to_round = ['Annual_Income', 'Changed_Credit_Limit', 'Monthly_Balance', 'Amount_invested_monthly',
                        'Total_EMI_per_month', 'Credit_Utilization_Ratio']  # , 'Monthly_Inhand_Salary'
    for column in columns_to_round:
        df[column] = pd.to_numeric(df[column], errors='coerce').round(2)

    # Fill Gaps in 'Credit_History_Age_by_year'
    valid_occupation_mapping = df.loc[df['Credit_History_Age_by_year'].notna()].groupby('Customer_ID')[
        'Credit_History_Age_by_year'].first()
    df['Credit_History_Age_by_year'] = df.apply(
        lambda row: valid_occupation_mapping.get(row['Customer_ID'], row['Credit_History_Age_by_year'])
        if pd.isna(row['Credit_History_Age_by_year'])
        else row['Credit_History_Age_by_year'],
        axis=1
    )

    # Clean Occupation '_______'
    valid_occupation_mapping = df.loc[df['Occupation'] != "_______"].groupby('Customer_ID')['Occupation'].first()
    df['Occupation'] = df.apply(
        lambda row: valid_occupation_mapping.get(row['Customer_ID'], "_______")
        if row['Occupation'] == "_______"
        else row['Occupation'],
        axis=1
    )
    # Replace values in 'Occupation'
    df['Occupation'] = df['Occupation'].map({
        "Accountant": 1,
        "Architect": 2,
        "Developer": 3,
        "Doctor": 4,
        "Engineer": 5,
        "Entrepreneur": 6,
        "Journalist": 7,
        "Lawyer": 8,
        "Manager": 9,
        "Mechanic": 10,
        "Media_Manager": 11,
        "Musician": 12,
        "Scientist": 13,
        "Teacher": 14,
        "Writer": 15
    })

    # Clean and Replace values in 'Payment_Behaviour'
    df = df[df['Payment_Behaviour'] != "!@9#%8"]
    df['Payment_Behaviour'] = df['Payment_Behaviour'].map({
        "High_spent_Large_value_payments": 1,
        "High_spent_Medium_value_payments": 2,
        "High_spent_Small_value_payments": 3,
        "Low_spent_Large_value_payments": 4,
        "Low_spent_Medium_value_payments": 5,
        "Low_spent_Small_value_payments": 6
    })

    # Clean Age
    valid_age_mapping = df.loc[df['Age'] <= 80].groupby('Customer_ID')['Age'].min()
    df['Age'] = df.apply(
        lambda row: valid_age_mapping.get(row['Customer_ID'], row['Age'])
        if row['Age'] > 80 else row['Age'],
        axis=1
    )
    df = df[(df['Age'] >= 18) & (df['Age'] <= 130)]

    # Remove empty rows
    # df = df.dropna(how='any')

    # Type_of_Loan into columns, 1 means value is true and 0 that is false, empty means "No Loan" is true
    loan_types = [
        "No Loan", "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
        "Home Equity Loan", "Mortgage Loan", "Not Specified", "Payday Loan",
        "Personal Loan", "Student Loan"]
    df['Type_of_Loan'] = df['Type_of_Loan'].fillna("No Loan")
    for loan in loan_types:
        df[loan] = df['Type_of_Loan'].str.contains(loan, case=False, na=False).astype(int)

    # Remove Columns
    columns_to_drop = ['ID', 'Name', 'SSN', 'Monthly_Inhand_Salary', 'Credit_History_Age', 'Customer_ID', 'Credit_Mix',
                       'Type_of_Loan']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Save the preprocessed file
    df.to_csv(output_file, index=False)
    print('finish')


def plot(df, columns):
    for column in columns:
        if column in df.columns:
            plt.figure(figsize=(10, 2))
            plt.boxplot(df[column].dropna(), vert=False, patch_artist=True)
            plt.title(f'Box Plot - {column}')
            plt.xlabel(column)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Column '{column}' not found in DataFrame.")


# File Path
# current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = "data/test.csv"
output_file = "data/prepared_test.csv"
preprocess_csv(input_file, output_file)

print("Ready!")

df = pd.read_csv(output_file)
total_count = df['Interest_Rate'].shape[0]
count_greater_than_100 = df[df['Interest_Rate'] > 40].shape[0]
print(f'Total count is: {total_count} and greater than 40 is {count_greater_than_100}')

df= pd.read_csv('data/prepared_test.csv')

sampled1 = df[df['Binary_credit_mix'] == 1].sample(n=5000, random_state=1)
sampled2 = df[df['Binary_credit_mix'] == 0].sample(n=5000, random_state=1)  # Sampling from class 0

# Merge results (balanced dataset)
result_df = pd.concat([sampled1, sampled2]).reset_index(drop=True)

dataframe1 = result_df.dropna().copy()

###############################

target_col = 'Binary_credit_mix'
input_cols = ['Month', 'Occupation']

'''input_cols = ['Month', 'Age', 'Occupation', 'Annual_Income', 'Num_Bank_Accounts',
              'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
              'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
              'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
              'Payment_of_Min_Amount', 'Total_EMI_per_month',
              'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
              'Customer_ID_Number', 'Credit_History_Age_by_year',
              'No Loan', 'Auto Loan', 'Credit-Builder Loan',
              'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan',
              'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']'''


print("dataframe1.columns = %r" % (dataframe1.columns,))
N = dataframe1.shape[0]
# if no target name is supplied we assume it is the last colunmn in the
# data file
if target_col is None:
    target_col = dataframe1.columns[-1]
    potential_inputs = dataframe1.columns[:-1]
else:
    potential_inputs = list(dataframe1.columns)
    # target data should not be part of the inputs
    potential_inputs.remove(target_col)
# if no input names are supplied then use them all
if input_cols is None:
    input_cols = potential_inputs
print("input_cols = %r" % (input_cols,))

classes = [0, 1]
if classes is None:
    # get the class values as a pandas Series object
    class_values = dataframe1[target_col]
    classes = class_values.unique()
else:
    # construct a 1d array of the rows to keep
    to_keep = np.zeros(N, dtype=bool)
    for class_name in classes:
        to_keep |= (dataframe1[target_col] == class_name)
    # now keep only these rows
    dataframe1 = dataframe1[to_keep]
    # there are a different number of dat items now
    N = dataframe1.shape[0]
    # get the class values as a pandas Series object
    class_values = dataframe1[target_col]

targets = np.empty(N)
for class_id, class_name in enumerate(classes):
    is_class = (class_values == class_name)
    targets[is_class] = class_id
# print("targets = %r" % (targets,))

# We're going to assume that all our inputs are real numbers (or can be
# represented as such), so we'll convert all these columns to a 2d numpy
# array object
inputs = dataframe1[input_cols].values
print('Return:', inputs, targets, input_cols, classes)


def fit_and_plot_roc_logistic_regression(
        inputs, targets, fig_ax=None, colour=None):
    """
    Takes input and target data for classification and fits shared covariance
    model. Then plots the ROC corresponding to the fit model.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = logistic_regression_fit(inputs, targets)
    #
    thresholds = np.linspace(0, 1, 101)
    N = targets.size
    num_neg = np.sum(1 - targets)
    num_pos = np.sum(targets)

    false_positive_rates = np.empty(thresholds.size)
    true_positive_rates = np.empty(thresholds.size)
    for i, threshold in enumerate(thresholds):
        prediction_probs = logistic_regression_prediction_probs(inputs, weights)
        predicts = (prediction_probs > threshold).astype(int)
        # print('predicts:',predicts)
        num_false_positives = np.sum((predicts == 1) & (targets == 0))
        num_true_positives = np.sum((predicts == 1) & (targets == 1))
        false_positive_rates[i] = np.sum(num_false_positives) / num_neg
        true_positive_rates[i] = np.sum(num_true_positives) / num_pos
    # print(false_positive_rates, true_positive_rates)
    fig, ax = plot_roc(
        false_positive_rates, true_positive_rates, fig_ax=fig_ax, colour=colour)
    # and for the class prior we learnt from the model
    predicts = logistic_regression_predict(inputs, weights)
    fpr = np.sum((predicts == 1) & (targets == 0)) / num_neg
    tpr = np.sum((predicts == 1) & (targets == 1)) / num_pos
    ax.plot([fpr], [tpr], 'rx', markersize=8, markeredgewidth=2)
    return fig, ax, weights


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig_ax = (fig, ax)
# we plot both the generative and logistic regression models on the same ROC curve

_, _, weights = fit_and_plot_roc_logistic_regression(
    inputs, targets, fig_ax=fig_ax, colour='b')

def get_acc(pred,target):
    correct_predictions = np.sum(pred == target)  # Number of correct predictions
    total_predictions = target.size  # Number of total predictions
    accuracy = correct_predictions / total_predictions
    return accuracy

predicts = logistic_regression_predict(inputs, weights)

print('The accuraacy of logistic regression is:',get_acc(predicts,targets))