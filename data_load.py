# Create a pandas DataFrame from the iris.csv file (sourced from the Seaborn Repository linked above).
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Write this DataFrame to our working directory 
df.to_csv('iris.csv')

# Let's have an initial look. 
df

# Let's generate some general information on the data
df.info()

