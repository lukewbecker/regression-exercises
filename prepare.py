# Prep iris database script for ingesting, cleaning and preparing the iris database for exploration.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prep_iris(x): 
    x.rename(columns = {'species_name': 'species'}, inplace=True)
    x = x.drop(columns = ['species_id', 'measurement_id'])
    dummy_df = pd.get_dummies(x['species'], dummy_na=False)
    x = pd.concat([x, dummy_df], axis=1)
    return x



def prep_titanic_data(titanic_data):
    # Importing the libraries I'll need for this function.
    from sklearn.impute import SimpleImputer
    import warnings
    warnings.filterwarnings("ignore")
    
    # Handling the missing data
    titanic_data = titanic_data[~titanic_data.embark_town.isnull()]
    # Removing the 'deck' column    
    titanic_data = titanic_data.drop(columns = 'deck')
    # Creating dummy variables
    dummy_titanic_df = pd.get_dummies(titanic_data['embarked'], dummy_na = False)
    titanic_data = pd.concat([titanic_data, dummy_titanic_df], axis=1)
    
    # Using the impute method to fill the missing values in the age column
    imputer = SimpleImputer(strategy = 'most_frequent')
    imputer.fit(titanic_data[['age']])
    titanic_data[['age']] = imputer.transform(titanic_data[['age']])
    return titanic_data

# Prep file for mall data:

def prep_mall_data(df):
    from sklearn.model_selection import train_test_split
    '''
    Takes the acquired data, does data prep, and returns
    train, test, and validate data splits
    '''
    df['is_female'] = (df.gender == 'Female').astype(int)
    train_validate, test = train_test_split(df, test_size = .15, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .15, random_state = 123)
    return train, validate, test
    print('imported mall prep function successfully')

print('Imported prepare.py successfully')
'End of file'