# Import pandas, matplotlib.pyplot and numpy packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a pandas DataFrame from a csv (providing url of the raw data)
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Write the DataFrame as a csv so we have a local copy
df.to_csv('iris.csv')

# https://note.nkmk.me/en/python-pandas-agg-aggregate/
#print(df.select_dtypes(include='number').agg

# Describe the data set and print the output to 'variable_summary.csv'
# https://practicaldatascience.co.uk/data-science/how-to-create-descriptive-statistics-using-the-pandas-describe-function
df.describe(include='all').T.to_csv('variable_summary.csv')


# https://www.geeksforgeeks.org/loop-or-iterate-over-all-or-certain-columns-of-a-dataframe-in-python-pandas/
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.items.html

