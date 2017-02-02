import pandas as pd
import os


def build_labels:
    '''
    Builds default loan model's classification labels, i.e. dependent variable

    Choosing a binary classification type label where a loan_status of Fully Paid is 0 and Default is 1

    Ignoring Current because outcome of the loan is not known yet and this model is focused
    on the final outcome of loan if it was given to an applicant, i.e. going to default (positive label) or fully pay (negative label)
    '''
    raw_data_df = pd.read_csv('data/data.csv')


def build_features:
    '''
    '''
    pass



if __name__ == '__main__':
    build_labels()