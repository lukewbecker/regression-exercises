# This function section is specifically designed to create visualizations for the explore phase of the data science pipeline process.
# These functions are specifically tailored for the telco dataset.

# importing libraries needed:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math




# Creating the function to plot a series of pairwise relationships:

def plot_variable_pairs(df):
    graph = sns.PairGrid(df) 
    graph.map_diag(plt.hist) 
    graph.map_offdiag(sns.regplot) 

# Creating a function to change the tenure column (which is in months) into years. Note that this function will round down to zero, so in essence the years are a categorical value.

def months_to_years(df):
    df.assign(tenure_years = (df.tenure / 12).apply(math.floor))
    return df

# Creating the function months to years (using Group 2 [my team] method):

def months_to_years_mine(tenure_months, df):
    df['tenure_years'] = round(tenure_months // 12, 0)
    return df

# This function will return 3 charts based on the telco data

def plot_categorical_and_continuous_vars(cat_var, cont_var, df):
    '''
    This function will take in a categorical variable and a continuous variable
    and return 3 charts based on those variables.
    Note that the categorical variable is tied to the x axis, and the continuious variable to the y axis.
    ''' 
    # Add in an if function that returns an error if the categorical variable isn't a categorical type(?)
    # That may not be useful...
    
    
    sns.barplot(data = df, y = cont_var, x = cat_var)
    plt.show()
    sns.violinplot(data = df, y = cont_var, x = cat_var)
    plt.show()
    sns.boxplot(data = df, y = cont_var, x = cat_var)

print('Functions loaded properly')
    