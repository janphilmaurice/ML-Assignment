import pandas as pd
import matplotlib.pyplot as plt


def preprocess_csv(input_file, output_file):
    # Load the data from the input CSV file
    df = pd.read_csv(input_file).dropna()
    # print(df.isnull().any())

    # Create a 'Customer_ID_Number'
    unique_customer_mapping = {customer_id: idx + 1 for idx, customer_id in enumerate(df['Customer_ID'].unique())}
    df['Customer_ID_Number'] = df['Customer_ID'].map(unique_customer_mapping)

    # Remove rows where 'Credit_Mix' & 'Changed_Credit_Limit' have the value "-"
    df = df[df['Credit_Mix'] != "_"]
    df = df[df['Changed_Credit_Limit'] != "_"]

    # Create a new column 'Binary_credit_mix' and 'Credit_History_Age_by_year'
    # df['Binary_credit_mix'] = df['Credit_Mix'].apply(lambda x: "1" if x == "Good" else "0")
    df['Binary_credit_mix'] = df['Credit_Mix'].map({
        "Good": 0,
        "Standard": 1,
        "Bad": 2
    })
    df = df[df['Credit_Score'] != "_"]
    #df = df[df['Credit_Score'] != "Standard"]
    df['Binary_Credit_Score'] = df['Credit_Score'].apply(lambda x: 1 if x == "Good" else 0)

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
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12

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
    ).astype(int)

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

    df['Occupation'] = df['Occupation'].fillna(df['Occupation'].mean())
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].fillna(
        df['Amount_invested_monthly'].mean())
    df['Monthly_Balance'] = df['Monthly_Balance'].fillna(
        df['Monthly_Balance'].mean())

    # Remove Columns
    columns_to_drop = ['ID', 'Name', 'SSN', 'Monthly_Inhand_Salary', 'Credit_History_Age', 'Customer_ID', 'Credit_Mix',
                       'Type_of_Loan', 'Credit_Score']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Save the preprocessed file
    df.to_csv(output_file, index=False)
    return df


def split_Xy(data,column):
    X = data.loc[:, data.columns != column].values  # inputs
    y = data[column]
    return X, y