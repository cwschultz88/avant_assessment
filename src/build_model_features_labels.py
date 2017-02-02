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
    Builds default loan model's features

    Saves results in:
        data/model_features.csv
    '''
    loan_data_df = pd.read_csv('data/data.csv').set_index("id") # load in data
    loan_data_df = loan_data_df[loan_data_df['loan_status'] != 'Current'] # remove current status entries

    # grab columns from dataframe that will be turned into features for this model
    columns_to_exclude = ['loan_status', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low']
    for column in columns_to_exclude:
        loan_data_df = loan_data_df.drop(column, 1)

    # replace issue_d with just the month int
    convert_month_to_int = {'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4, 'Jun':5, 'Jul':6, 'Aug':7, 'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}
    issue_months = [convert_month_to_int[value[:3]] for value in loan_data_df.ix[:,'issue_d'].values]
    loan_data_df['issue_d'] = issue_months

    # for columns don't care about how str categories are turned into ints, perform simple int encoding
    columns_simple_int_encodings = ['term', 'home_ownership', 'verification_status', 'purpose', 'addr_state']
    for column in columns_simple_int_encodings:
            unique_values = loan_data_df[column].unique()
            str_encoding_mapping = {unique_value:i for i,unique_value in enumerate(unique_values)}
            loan_data_df[column].replace(str_encoding_mapping, inplace=True)

    # employment str to int encoding
    employment_encodings = {'n/a':-1, '< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}
    loan_data_df['emp_length'].replace(employment_encodings, inplace=True)

    # replace earliest_cr_line with 2017 - year of credit open
    years_difference = [2017 - int(value[4:]) for value in loan_data_df.ix[:,'earliest_cr_line'].values]
    loan_data_df['earliest_cr_line'] = years_difference

    # replace NaNs with -1
    loan_data_df = loan_data_df.fillna(-1)

    loan_data_df.to_csv('data/model_features.csv')


if __name__ == '__main__':
    build_labels()
    build_features()
