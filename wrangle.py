# File for creating an acquire and prepping of files for regression exercises.

# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os

# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

# split_scale
# import split_scale

# libraries needed for preparing the data:
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
# from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing

# Setting up the user credentials:
from env import host, user, password


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}' 

# Acquire the telco_churn data from the Codeup DB.

# There are two different acquire functions contained in here because the exercises called for a specific set of data points, 
# which implied that the sorting should occur within SQL. Hence the two separate acquire functions.

def telco_data_two_year():
    '''
    This function reads the telco customer data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    This function was written specifically to match the requirements in the text,
    which appears to dictate that the filtering of contract type must happen in the acquire stage.
    The only contract type returned will be the two-year contract from the telco db.
    '''
    sql_query_two_year = '''
                        SELECT c.customer_id, c.monthly_charges, c.tenure, c.total_charges, c.contract_type_id, ct.contract_type
                        FROM customers as c
                        JOIN payment_types as pt on c.payment_type_id = pt.payment_type_id
                        JOIN internet_service_types AS ist on ist.internet_service_type_id = c.internet_service_type_id
                        JOIN contract_types as ct ON ct.contract_type_id = c.contract_type_id
                        WHERE c.contract_type_id = 3;
                        '''
        
    df = pd.read_sql(sql_query_two_year, get_connection('telco_churn'))
    df.to_csv('telco_customers_df_two_year.csv')
    return df

# def get_telco_data_two():
#         '''
#         This function reads in mall customer data from Codeup database if cached == False 
#         or if cached == True reads in telco customer df from a csv file, returns df
#         '''
#     if cached or os.path.isfile('telco_customers_df_two_year.csv') == False:
#         df = telco_two_year()
#     else:
#         df = pd.read_csv('telco_customers_df_two_year.csv', index_col=0)
#     return df


def get_telco_data_two_year(cached = False):
    '''
    This function reads in mall customer data from Codeup database if cached == False 
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('telco_customers_df_two_year.csv') == False:
        df = telco_data_two_year()
    else:
        df = pd.read_csv('telco_customers_df_two_year.csv', index_col=0)
    return df



# Adding the scaled columns to the telco dataframe:

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)

    return train, validate, test



# Adding the scaled data to the dataframe, returning the train, validate and test dataframes.

def scale_telco_data(train, test, validate):
    from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

    train, validate, test = add_scaled_columns(
        train,
        test,
        validate,
        scaler = sklearn.preprocessing.MinMaxScaler(),
        columns_to_scale=['total_charges', 'monthly_charges', 'tenure'],
    )
    return train, validate, test


# Preparing the data:
# The prep function returning the train, validate and test splits:
def prep_acquired_telco():
    '''
    This function will both acquire and prep the modified telco dataset (only 2-yr contracts)
    and return the train, validate, and test datasets. It will read off a .csv that is in the working directory
    if it exists, otherwise the function will pull the data from the Codeup db.
    '''
    # First, I need to acquire the dataframe within this function:
    df = pd.read_csv('telco_customers_df_two_year.csv', index_col = 0)

    
    # Cleaning the total_costs column by dropping empty values:
    df.drop(df[df['total_charges'] == " "].index, inplace = True)
    
    # Changing the total_costs column to a float:
    df['total_charges'] = df.total_charges.astype('float')
    
    # Finally, splitting my data based on the target variable of tenure:
    
    train_validate, test = train_test_split(df, test_size=.15, 
                                            random_state=123)
    
    # Splitting the train_validate set into the separate train and validate datasets.
    train, validate = train_test_split(train_validate, test_size=.15, 
                                   random_state=123)
    print(f'Shape of train df: {train.shape}')
    print(f'Shape of validate df: {validate.shape}')
    print(f'Shape of test df: {test.shape}')
    return scale_telco_data(train, validate, test)


# Function that scales data:

def telco_scaled_data(train, validate, test):
    
    from sklearn.preprocessing import StandardScaler
    
    # 1. Creating the scaler object:

    scaler_ss = sklearn.preprocessing.StandardScaler()

    # 2. Fitting ONLY do the train dataset. Remember, this is only useful on continuous variables, not categorical variables:
    scaler_ss.fit(train[['total_charges', 'monthly_charges', 'tenure']])

    # 3. Now use the object on all three datasets. Remember, I'm using this object which was fitted to the train dataset:
    train[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler_ss.transform(train[['total_charges', 'monthly_charges', 'tenure']])
    validate[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler_ss.transform(validate[['total_charges', 'monthly_charges', 'tenure']])
    test[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler_ss.transform(test[['total_charges', 'monthly_charges', 'tenure']])

    return train, validate, test


######################### Complete Wrangle for Telco #########################
def wrangle_telco():
    '''
    This function will read the telco data from a csv file or the codeup db,
    preps the columns called by the two year telco function (total_charges changed to int),
    and returns the split train, validate and test dataframes
    '''
    df = get_telco_data_two_year()

    # Cleaning the total_costs column by dropping empty values:
    df.tenure.replace(0, 1, inplace=True)
    df.total_charges = df.total_charges.replace(" ", np.nan)
    df.total_charges = df.total_charges.fillna(df.monthly_charges)
    df.total_charges = df.total_charges.astype(float)
    
    # Finally, splitting my data based on the target variable of tenure:
    train_validate, test = train_test_split(df, test_size=.15, random_state=123)
    
    # Splitting the train_validate set into the separate train and validate datasets.
    train, validate = train_test_split(train_validate, test_size=.15, random_state=123)
    
    # Printing the shape of each dataframe:
    print(f'Shape of train df: {train.shape}')
    print(f'Shape of validate df: {validate.shape}')
    print(f'Shape of test df: {test.shape}')
    return scale_telco_data(train, validate, test)


# Grades wrangle (acquire and prep) function for explore lesson walkthrough:
def wrangle_grades():
    grades = pd.read_csv("student_grades.csv")
    grades.drop(columns="student_id", inplace=True)
    grades.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df = grades.dropna().astype("int")
    return df


print('wrangle.py functions loaded successfully.')