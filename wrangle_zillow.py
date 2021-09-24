import os
from env import host, user, password
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


################### Connects to Sequel Ace using credentials ###################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


################### Create new dataframe from SQL db ###################
    
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    writes it to a csv file, and returns the df.
    '''

    # Create SQL query.
    sql_query = """
            SELECT 	prop.*, 
                ac.airconditioningdesc,
                txn.transactiondate, 
                pred.logerror, 
                ast.architecturalstyledesc,
                bc.buildingclassdesc,
                hs.heatingorsystemdesc,
                plu.propertylandusedesc,
                st.storydesc,
                tc.typeconstructiondesc
            FROM properties_2017 prop
            JOIN (
                SELECT parcelid, max(transactiondate) as transactiondate
                FROM predictions_2017
                GROUP BY parcelid
                ) AS txn ON prop.parcelid = txn.parcelid
            JOIN predictions_2017 AS pred  
            ON prop.parcelid = pred.parcelid 
                AND pred.transactiondate = txn.transactiondate
            LEFT JOIN airconditioningtype AS ac
            USING(airconditioningtypeid)
            LEFT JOIN architecturalstyletype AS ast
            USING (architecturalstyletypeid)
            LEFT JOIN buildingclasstype AS bc
            USING (buildingclasstypeid)
            LEFT JOIN heatingorsystemtype AS hs
            USING (heatingorsystemtypeid)
            LEFT JOIN propertylandusetype AS plu
            USING (propertylandusetypeid)
            LEFT JOIN storytype AS st
            USING (storytypeid)
            LEFT JOIN typeconstructiontype AS tc
            USING (typeconstructiontypeid)
            WHERE COALESCE(prop.longitude, prop.latitude) IS NOT NULL;
                """
    # Read in DataFrame from Codeup's SQL db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df


################### Acquire existing csv file ###################

def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df


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



def viz_outliers(df, k, col_to_viz):
    for col in col_to_viz:
        if df[col].dtype != 'O':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound =  q3 + k * iqr
            lower_bound =  q1 - k * iqr
            ax=sns.distplot(df[col])
            plt.axvline(lower_bound)
            plt.axvline(upper_bound)
            plt.title(f'Distribution of {col} with \n upper and lower bounds')
            plt.show()
            
            
            
def remove_outliers(df, cols, k):
    for col in df[cols]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df
                                   
                                   
                                   
def min_max_scaler(train, validate, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    
    train_mm = train.copy()
    validate_mm = validate.copy()
    test_mm = test.copy()
    
    num_vars = list(train_mm.select_dtypes('number').columns)
    scaler = MinMaxScaler()
    train_mm[num_vars] = scaler.fit_transform(train_mm[num_vars])
    validate_mm[num_vars] = scaler.transform(validate_mm[num_vars])
    test_mm[num_vars] = scaler.transform(test_mm[num_vars])
    return scaler, train_mm, validate_mm, test_mm
                                   
                                   
                          
def train_validate_test_split(df):
    train_and_validate, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_and_validate, train_size=0.75, random_state=123)
    return train, validate, test