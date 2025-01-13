# Data Preprocessing

The original file is **Master_Data.csv**, and the preprocessing focuses on three main tasks:  
1. Creating the binary dataset  
2. Converting string-type variables into numerical values  
3. Eliminating data errors and unwanted columns  

All the main tasks are handled within the function `def preprocess_csv`.

The original dataset contains **27 columns** and **50,000 rows**. It consists of credit information spread over a 4-month period. Each individual is identified by a unique identifier (`Customer_ID`), and their personal information is displayed along with details such as credits taken, interest rates, payments, prepayments, etc.

---

## 1. Creating the Binary Dataset
The variables `Credit_Mix` and `Type_of_Loan` were converted into binary columns.

---

## 2. Converting String-Type Variables into Numbers
The following variables were transformed:  
- `Customer_ID` now renamed to `Customer_ID_Number`  
- `Month`
- `Payment_of_Min_Amount`
- `Occupation`
- `Payment_Behaviour`

---

## 3. Eliminating Data Errors and Unwanted Columns
The following redundant or unnecessary columns were removed:  
- `ID`  
- `Name`  
- `SSN`  
- `Monthly_Inhand_Salary`  
- `Credit_History_Age`  
- `Customer_ID` (original version)  
- `Credit_Mix` (original version)  
- `Type_of_Loan` (original version)  

### Handling Data Errors
Data errors were mostly in the form of typos and blank values. Information provided by the credit solicitor was filled in if rows were already complete; otherwise, incomplete rows were removed.

### Narrowing Variables
The following variables were refined using logical constraints:  
- `Age`
- `Annual_Income`
- `Outstanding_Debt`
- `Num_of_Delayed_Payment`
- `Num_of_Loan`

# Derived Datasets:

Pending to complete. Important because we need to decide how are we going to construct the derived datasets and explain the approach (in the report)