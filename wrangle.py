# File for creating an acquire and prepping of files for regression exercises.

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env


def prep_mall_data(df):
    '''
    Takes the acquired data, does data prep, and returns
    train, test, and validate data splits
    '''
    df['is_female'] = (df.gender == 'Female').astype(int)
    train_validate, test = train_test_split(df, test_size = .15, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .15, random_state = 123)
    return train, validate, test
