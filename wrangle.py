# File for creating an acquire and prepping of files for regression exercises.

# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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


# Function that scales data:

def telco_scaled_data(train, validate, test):
    
    from sklearn.preprocessing import MinMaxScaler
    
    # 1. Creating the scaler object:

    scaler = sklearn.preprocessing.MinMaxScaler()

    # 2. Fitting ONLY do the train dataset. Remember, this is only useful on continuous variables, not categorical variables:
    scaler.fit(train[['total_charges', 'monthly_charges', 'tenure']])

    # 3. Now use the object on all three datasets. Remember, I'm using this object which was fitted to the train dataset:
    train[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler.transform(train[['total_charges', 'monthly_charges', 'tenure']])
    validate[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler.transform(validate[['total_charges', 'monthly_charges', 'tenure']])
    test[['total_charges_scaled', 'monthly_charges_scaled', 'tenure_scaled']] = scaler.transform(test[['total_charges', 'monthly_charges', 'tenure']])

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


# Wrangling mall_customer database:
def get_mall_data():
    sql_query = '''
            SELECT *
            FROM customers
            '''

    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    return df


# Prepping the mall data:

def wrangle_mall_data():
    df = get_mall_data()
    # Splitting my data based on the target variable of tenure:
    train_validate, test = train_test_split(df, test_size=.15, random_state=123)
    
    # Splitting the train_validate set into the separate train and validate datasets.
    train, validate = train_test_split(train_validate, test_size=.20, random_state=123)
    
    # Printing the shape of each dataframe:
    print(f'Shape of train df: {train.shape}')
    print(f'Shape of validate df: {validate.shape}')
    print(f'Shape of test df: {test.shape}')
    return train, validate, test

def split_student_data(df):
    # df = get_mall_data()
    # Splitting my data based on the target variable of tenure:
    train_validate, test = train_test_split(df, test_size=.15, random_state=123)
    
    # Splitting the train_validate set into the separate train and validate datasets.
    train, validate = train_test_split(train_validate, test_size=.20, random_state=123)
    
    # Printing the shape of each dataframe:
    print(f'Shape of train df: {train.shape}')
    print(f'Shape of validate df: {validate.shape}')
    print(f'Shape of test df: {test.shape}')
    return train, validate, test

###################### These functions are to prep the data and return the scaled X and y train, validate, and test dataframes #################

def wrangle_student_math(path):
    df = pd.read_csv(path, sep=";")
    
    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)
    
    # create dummy vars
    df = create_dummies(df, object_cols)
      
    # split data 
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, 'G3')
    
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    return df, X_train, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test


# functions from class:

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols
    
def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df


##### This is the key function that returns 6 dataframes #####
def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled


def min_max_scale(X_train, X_validate, X_test):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).
    
    scaler = MinMaxScaler(copy = True).fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns.values).set_index([X_train.index.values])
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns = X_validate.columns.values).set_index([X_validate.index.values])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns.values).set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled




def wrangle_telco_two_year(path):
    df = pd.read_csv(path, index_col = 0)
    
    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)
    
    # create dummy vars
    df = create_dummies(df, object_cols)
      
    # split data 
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, 'tenure')
    
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    return df, X_train, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test

def wrangle_telco(scale_data =True):
    ''' 
    This function preforms 4 operations:
    1. Reads in the telco data from a csv file
    2. Changes total_charges to a numeric variable and replaces any NaN values with a 0 
    3. Splits prepared data in to train, validate, test  
    4. If scale_data = True returns scaled train, validate, test
       If scale_data = False returns non scaled train, validate, test
    '''
    df = get_telco_data(cached = False)
    # Changes total_charges to numeric variable
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    # Replaces NaN values with 0 for new customers with no total_charges
    df["total_charges"].fillna(0, inplace = True) 
    # Split the data in to train, validate, test
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    if scale_data:
        # Inverse scale
        #train, validate, test = scale_inverse(train, validate, test)
        #Scales data and return scaled data dataframes
        return scale_telco_data(train, validate, test)
    return train, validate, test


#### Complete Telco Wrangle ####

def wrangle_telco_data_all(path):
    # First, I need to acquire the dataframe within this function:
    df = pd.read_csv('telco_customers_df_two_year.csv', index_col = 0)

    # Cleaning the total_costs column by dropping empty values:
    df.drop(df[df['total_charges'] == " "].index, inplace = True)

    # Changing the total_costs column to a float:
    df['total_charges'] = df.total_charges.astype('float')

    # get object column names
    object_cols = get_object_cols(df)
    
    # create dummy vars
    df = create_dummies(df, object_cols)
      
    # split data 
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, 'total_charges')
    
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    return df, X_train, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test


def wrangle_telco_data_all_target(path, target):
    # First, I need to acquire the dataframe within this function:
    df = pd.read_csv('telco_customers_df_two_year.csv', index_col = 0)

    # Cleaning the total_costs column by dropping empty values:
    df.drop(df[df['total_charges'] == " "].index, inplace = True)

    # Changing the total_costs column to a float:
    df['total_charges'] = df.total_charges.astype('float')

    # get object column names
    object_cols = get_object_cols(df)
    
    # create dummy vars
    df = create_dummies(df, object_cols)
      
    # split data 
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, target)
    
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    return df, X_train, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test

print('wrangle.py functions loaded successfully.')