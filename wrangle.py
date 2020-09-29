# File for creating an acquire and prepping of files for regression exercises.

# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import os

# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

# libraries possibly needed for preparing the data:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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


def get_telco_data_two_year(cached=False):
    '''
    This function reads in mall customer data from Codeup database if cached == False 
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('telco_customers_df_two_year.csv') == False:
        df = telco_data_two_year()
    else:
        df = pd.read_csv('telco_customers_df_two_year.csv', index_col=0)
    return df


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
    return telco_scaled_data(train, validate, test)


# Function that scales data:

def telco_scaled_data(train, validate, test):
    # 1. Creating the scaler object:

    scaler_ss = sklearn.preprocessing.StandardScaler()

    # 2. Fitting ONLY do the train dataset. Remember, this is only useful on continuous variables, not categorical variables:
    scaler_ss.fit(train[['total_charges', 'monthly_charges', 'tenure']])

    # 3. Now use the object on all three datasets. Remember, I'm using this object which was fitted to the train dataset:
    train[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler_ss.transform(train[['total_charges', 'monthly_charges', 'tenure']])
    validate[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler_ss.transform(validate[['total_charges', 'monthly_charges', 'tenure']])
    test[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler_ss.transform(test[['total_charges', 'monthly_charges', 'tenure']])

    return train, validate, test


# Grades wrangle (acquire and prep) function:
def wrangle_grades():
    grades = pd.read_csv("student_grades.csv")
    grades.drop(columns="student_id", inplace=True)
    grades.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df = grades.dropna().astype("int")
    return df


print('wrangle.py functions loaded successfully.')