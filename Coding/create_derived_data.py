import pandas as pd

def split_data(data,column,rate=(0.5,0.5),num=5000):
    num1 = int(num*rate[0])
    num2 = int(num*rate[1])
    #print(data[data['Binary_Credit_Score'] == 0])
    sampled1 = data[data[column] == 0].sample(n=num1, random_state=1)
    sampled2 = data[data[column] == 1].sample(n=num2, random_state=1)  # sample for class1


    # Combined result
    result_df = pd.concat([sampled1, sampled2]).reset_index(drop=True)
    return result_df