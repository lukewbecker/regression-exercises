# Creating the aquire.py file

# Make a new python module, acquire.py to hold the following data aquisition functions:
# get_titanic_data
# get_iris_data

# Importing libraries:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Importing the os library specifically for reading the csv once I've created the file in my working directory.
import os

# Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame. Obtain your data from the Codeup Data Science Database.

# Setting up the user credentials:

from env import host, user, password

def get_db(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Telco acquire function
# Being explicit in my SQL query allows me to only pull the extra info from the columns in the other tabes that I actually need, instead of returning the repeated foreign key columns. Saves me time on the prepare step.
# This acquire function will allow the user, with proper credentials to the CodeUp database access the same original dataframe that I started with in my process.
# It will first search for a csv file containing the appropriate telco data in the same folder that the jupyter notebook is being run in, and if it doesn't find one
# the function will execute the MySQL query call and create a local copy of the telco dataset.

# Recall, the way to call this function specifically is to type the following:
#   from acquire.py import get_telco_data
# That'll import this function for use in your own notebook.

def get_telco_data():
    
    my_telco_query = '''SELECT c.*, pt.payment_type, ist.internet_service_type, ct.contract_type
                FROM customers as c
                JOIN payment_types as pt on c.payment_type_id = pt.payment_type_id
                JOIN internet_service_types AS ist on ist.internet_service_type_id = c.internet_service_type_id
                JOIN contract_types as ct ON ct.contract_type_id = c.contract_type_id;'''
    
    
    filename = 'telco_data.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        telco_df = pd.read_sql(my_telco_query, get_db('telco_churn'))
        telco_df.to_csv(filename, index = False)
        
    return telco_df


print('End of file.')