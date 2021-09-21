import os
from env import host, user, password
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def summarize_stats(df):
    print("YOU CAN'T HANDLE THE STATS!!!!!!")
    print('|------------------------------------------------------|')
    print('|------------------------------------------------------|')
    print(f'Shape: {df.shape}')
    print('|------------------------------------------------------|')
    print(df.info())
    print('|------------------------------------------------------|')
    print('|------------------------------------------------------|')
    for col in df.columns:
        print(f'|-------{col}-------|')
        print()
        print(f'dtpye: {df[col].dtype}')
        print()
        print(f'Null count: {df[col].isnull().sum()}')
        print()
        print(df[col].describe())
        print()
        print(df[col].value_counts())
        print()
        print(df[col].unique())
        print()
        if df[col].dtype != 'O':
            sns.distplot(df[col])
            plt.title(f'Distribution of {col}')
            plt.ylabel('Frequency')
            plt.xlabel(col)
            plt.show()
        print('|------------------------------------------------------|')
        print('|------------------------------------------------------|')



def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing



def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing



def single_unit_props(df):
    df = df[(df.propertylandusetypeid==261) | 
            (df.propertylandusetypeid==263) | 
            (df.propertylandusetypeid==264) | 
            (df.propertylandusetypeid==265) | 
            (df.propertylandusetypeid==266) | 
            (df.propertylandusetypeid==275) | 
            (df.unitcnt==1)]
    return df



def handle_missing_values(df, prop_required_column=0.75, prop_required_row=0.5):
    df.dropna(axis=0, thresh=df.shape[1]*prop_required_row, inplace=True)
    df.dropna(axis=1, thresh=df.shape[0]*prop_required_column, inplace=True)
    return df



def drop_useless(df, useless_cols):
    df.drop(columns=useless_cols, inplace=True)
    return df



def cols_missing_10000(df, new_col_thresh=10_000):
    remove_cols=[]
    for col in df.columns:
        if df[col].isnull().sum() > new_col_thresh:
            remove_cols.append(col)
    df.drop(columns=remove_cols, inplace=True)
    return df



def impute_mode(df, cols_to_fill):
    for col in cols_to_fill:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df



def impute_median(df, cols_to_fill):
    for col in cols_to_fill:
        df[col].fillna(df[col].median(), inplace=True)
    return df