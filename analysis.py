# Import pandas, matplotlib.pyplot and numpy packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a pandas DataFrame based on the iris.csv file
df = pd.read_csv('iris.csv')

# Describe the data set and print the output to 'variable_summary.csv'
df.describe().to_csv('variable_summary.csv')

