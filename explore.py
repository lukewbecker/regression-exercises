# This function section is specifically designed to create visualizations for the explore phase of the data science pipeline process.
# These functions are specifically tailored for the telco dataset.

# importing libraries needed:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# Creating the function to plot a series of pairwise relationships:

def plot_variable_pairs(df):
    graph = sns.PairGrid(df) 
    graph.map_diag(plt.hist) 
    graph.map_offdiag(sns.regplot) 

# Creating a function to change the tenure column (which is in months) into years. Note that this function will round down to zero, so in essence the years are a categorical value.

def months_to_years(tenure_months, df):
    df['tenure_years'] = round(tenure_months / 12, 0)
    return df

# This function will return 3 charts based on the telco data

def plot_categorical_and_continuous_vars(cat_var, continuous_var,  df):
    '''
    This function will return 3 charts based on a categorical variable and a continuous variable. 
    Note that the categorical variable is tied to the x axis, and the continuious variable to the y axis.
    '''
    plt.rc('font', size = 13)
    plt.rc('figure', figsize = (13, 7))
    sns.boxplot(data = df, y = continuous_var, x = cat_var)
    plt.show()
    sns.swarmplot(data = df, y = continuous_var, x = cat_var)
    plt.show()
    sns.violinplot(data = df, y = continuous_var, x = cat_var)
    plt.show()

print('Functions loaded properly')
    