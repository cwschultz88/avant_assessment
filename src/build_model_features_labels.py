import pandas as pd
import os


def build_labels():
    '''
    Builds default loan model's classification labels, i.e. dependent variable

    Choosing a binary classification type label where a loan_status of Fully Paid is 0 and Default is 1

    Ignoring Current because outcome of the loan is not known yet and this model is focused
    on the final outcome of loan if it was given to an applicant, i.e. going to default (positive label) or fully pay (negative label)

    Saves results in:
        data/model_labels.csv
    '''
    loan_data_df = pd.read_csv('data/data.csv').set_index("id")
    loan_data_df = loan_data_df[loan_data_df['loan_status'] != 'Current']
    loan_data_df['loan_status'].replace({"Fully Paid":0, "Default":1}, inplace=True)
    loan_data_df.ix[:, ['loan_status']].to_csv('data/model_labels.csv')

def build_features():
    '''
    Saves results in:
        data/model_features.csv
    '''
    loan_data_df = pd.read_csv('data/data.csv').set_index("id") # load in data
    loan_data_df = loan_data_df[loan_data_df['loan_status'] != 'Current'] # remove current status entries
    loan_data_df = loan_data_df.drop('loan_status', 1)  # remove label

    # grab columns from dataframe that will be turned into features for this model
    columns_to_use_as_features = ['loan_amnt', 'term', 'installment', 'emp_length']



if __name__ == '__main__':
    build_labels()
    build_features()