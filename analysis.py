# Import packages required.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, accuracy_score


'''1. DATA LOAD AND OVERVIEW'''


# Create a pandas DataFrame from the iris.csv file (sourced from the Seaborn Repository linked above) and save a local copy.
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
df.to_csv('iris.csv')
df.info()


'''2. VARIABLE SUMMARY'''


# Split DataFrame into numeric and non-numeric sub Data Frames.
X = df.select_dtypes(float)
y = df.select_dtypes(object)

# Generate descriptive statistics for the Iris categorical column 'species'.
y_stats = y.describe().T
y_counts = y.value_counts()
y_stats.iat[0,2] = 'setosa, versicolor, virginica'

# Write summary of species vaeriable to csv file.
with open("variable_summary.csv", "w") as f:
    f.write("Summary of Iris categorical column below:" +"\n" +"\n")
y_stats.to_csv('variable_summary.csv', mode='a')

# Generate descriptive statistics for the Iris numerical columns.
X_stats = X.agg(['min', 'mean', 'median', 'std', 'max', 'skew', 'kurtosis']).T
X_stats['mean+3*std'] = X_stats['mean'] + X_stats['std'].mul(3)
X_stats['mean-3*std'] = X_stats['mean'] - X_stats['std'].mul(3)
X_stats = X_stats.reindex(columns=['mean-3*std', 'min', 'mean', 'median', 'max', 'mean+3*std', 'skew', 'kurtosis'])

#  Write summary of species vaeriable to csv file.
with open("variable_summary.csv", "a") as f:
    f.write("\n" + "Summary of Iris numerical columns below:" + "\n" +"\n")
X_stats.to_csv('variable_summary.csv', mode='a')


'''3. VISUALIZATIONS'''


####################
'''Iris_dist.png'''
####################

# Create a new figure and sets of axes, split into 2 rows and 2 columns.
fig, ax = plt.subplots(2,2)

# Set figure title & size.
fig.suptitle('Iris Variable Distributions', y=.925,fontweight = 900)
fig.set_figheight(10)
fig.set_figwidth(15)

# Plot histograms of the four variables.
ax[0,0].hist(df["sepal_length"], edgecolor='black', color="mediumorchid")
ax[0,1].hist(df["sepal_width"], edgecolor='black', color="mediumorchid")
ax[1,0].hist(df["petal_length"], edgecolor='black', color="mediumorchid")
ax[1,1].hist(df["petal_width"], edgecolor='black', color="mediumorchid")

# Set axis labels for four subplots.
ax[0,0].set_xlabel("Sepal Length (μ = 5.84)", fontweight = 550)
ax[0,0].set_ylabel("Count", fontweight = 550)
ax[0,1].set_xlabel("Sepal Width (μ = 3.06)", fontweight = 550)
ax[0,1].set_ylabel("Count", fontweight = 550)
ax[1,0].set_xlabel("Petal Length (μ = 3.76)", fontweight = 550)
ax[1,0].set_ylabel("Count", fontweight = 550)
ax[1,1].set_xlabel("Petal Width (μ = 1.2)", fontweight = 550)
ax[1,1].set_ylabel("Count", fontweight = 550)

# Save to png
plt.savefig('Iris_dist.png')

##############################
'''Iris_dist_by_species.png'''
##############################

# Create subset dataframes for the different species
setosa_df = df[df["species"].str.contains("setosa")]
versicolor_df = df[df["species"].str.contains("versicolor")]
virginica_df = df[df["species"].str.contains("virginica")]

# Create a new figure and sets of axes, split into 2 rows and 2 columns.
fig, ax = plt.subplots(2,2)

# Set figure title & size
fig.suptitle('Iris Variable Distributions (by Species)',y=.925,fontweight = 900)
fig.set_figheight(10)
fig.set_figwidth(15)

# Plot histograms of the four variables by species, each subplot has three histograms overlayed on each other.
# https://www.geeksforgeeks.org/overlapping-histograms-with-matplotlib-in-python/
ax[0,0].hist(setosa_df["sepal_length"], edgecolor='black', color="red", alpha=0.5)
ax[0,0].hist(versicolor_df["sepal_length"], edgecolor='black', color="green", alpha=0.5)
ax[0,0].hist(virginica_df["sepal_length"], edgecolor='black', color="blue", alpha=0.5)
ax[0,1].hist(setosa_df["sepal_width"], edgecolor='black', color="red", alpha=0.5)
ax[0,1].hist(versicolor_df["sepal_width"], edgecolor='black', color="green", alpha=0.5)
ax[0,1].hist(virginica_df["sepal_width"], edgecolor='black', color="blue", alpha=0.5)
ax[1,0].hist(setosa_df["petal_length"], edgecolor='black', color="red", alpha=0.5)
ax[1,0].hist(versicolor_df["petal_length"], edgecolor='black', color="green", alpha=0.5)
ax[1,0].hist(virginica_df["petal_length"], edgecolor='black', color="blue", alpha=0.5)
ax[1,1].hist(setosa_df["petal_width"], edgecolor='black', color="red", alpha=0.5)
ax[1,1].hist(versicolor_df["petal_width"], edgecolor='black', color="green", alpha=0.5)
ax[1,1].hist(virginica_df["petal_width"], edgecolor='black', color="blue", alpha=0.5)

# Set axis labels & legends for four subplots.
ax[0,0].set_xlabel("Sepal Length (μ = 5.84)", fontweight = 550)
ax[0,0].set_ylabel("Count", fontweight = 550)
ax[0,0].legend(['Setosa','Versicolor','Virginica'])
ax[0,1].set_xlabel("Sepal Width (μ = 3.06)", fontweight = 550)
ax[0,1].set_ylabel("Count", fontweight = 550)
ax[0,1].legend(['Setosa','Versicolor','Virginica'])
ax[1,0].set_xlabel("Petal Length (μ = 3.76)", fontweight = 550)
ax[1,0].set_ylabel("Count", fontweight = 550)
ax[1,0].legend(['Setosa','Versicolor','Virginica'])
ax[1,1].set_xlabel("Petal Width (μ = 1.2)", fontweight = 550)
ax[1,1].set_ylabel("Count", fontweight = 550)
ax[1,1].legend(['Setosa','Versicolor','Virginica'])

# Save to png
plt.savefig('Iris_dist_by_species.png')

##############################
'''Iris_Scatterplots_2D.png'''
##############################

# Create a new figure and sets of axes, split into 2 rows and 3 columns.
fig, ax = plt.subplots(2,3)

# Set figure title, size & label
fig.suptitle('Iris Scatterplots (Pairs)', y=.94, fontweight = 900)
fig.set_figheight(10)
fig.set_figwidth(15)

# Plot each pair of variables (6 pairs in total) against each other in scatterplots.

# First Pair: sepal length vs. sepal width
ax[0,0].scatter(setosa_df["sepal_length"], setosa_df["sepal_width"], color="red")
ax[0,0].scatter(versicolor_df["sepal_length"], versicolor_df["sepal_width"], color="green")
ax[0,0].scatter(virginica_df["sepal_length"], virginica_df["sepal_width"], color="blue")
ax[0,0].set_xlabel("Sepal Length (cm))", fontweight = 550)
ax[0,0].set_ylabel("Sepal Width (cm)", fontweight = 550)

# Second Pair: sepal length vs. petal length
ax[0,1].scatter(setosa_df["sepal_length"], setosa_df["petal_length"], color="red")
ax[0,1].scatter(versicolor_df["sepal_length"], versicolor_df["petal_length"], color="green")
ax[0,1].scatter(virginica_df["sepal_length"], virginica_df["petal_length"], color="blue")
ax[0,1].set_xlabel("Sepal Length (cm)", fontweight = 550)
ax[0,1].set_ylabel("Petal Length (cm)", fontweight = 550)

# Third Pair: sepal length vs. petal width
ax[0,2].scatter(setosa_df["sepal_length"], setosa_df["petal_width"], color="red")
ax[0,2].scatter(versicolor_df["sepal_length"], versicolor_df["petal_width"], color="green")
ax[0,2].scatter(virginica_df["sepal_length"], virginica_df["petal_width"], color="blue")
ax[0,2].set_xlabel("Sepal Length (cm)", fontweight = 550)
ax[0,2].set_ylabel("Petal Width (cm)", fontweight = 550)

# Fourth Pair: sepal width vs. petal length
ax[1,0].scatter(setosa_df["sepal_width"], setosa_df["petal_length"], color="red")
ax[1,0].scatter(versicolor_df["sepal_width"], versicolor_df["petal_length"], color="green")
ax[1,0].scatter(virginica_df["sepal_width"], virginica_df["petal_length"], color="blue")
ax[1,0].set_xlabel("Sepal Width (cm)", fontweight = 550)
ax[1,0].set_ylabel("Petal Length (cm)", fontweight = 550)

# Fifth Pair: sepal width vs. petal width
ax[1,1].scatter(setosa_df["sepal_width"], setosa_df["petal_width"], color="red")
ax[1,1].scatter(versicolor_df["sepal_width"], versicolor_df["petal_width"], color="green")
ax[1,1].scatter(virginica_df["sepal_width"], virginica_df["petal_width"], color="blue")
ax[1,1].set_xlabel("Sepal Width (cm)", fontweight = 550)
ax[1,1].set_ylabel("Petal Width (cm)", fontweight = 550)

# Sixth Pair: petal length vs. petal width
ax[1,2].scatter(setosa_df["petal_length"], setosa_df["petal_width"], color="red")
ax[1,2].scatter(versicolor_df["petal_length"], versicolor_df["petal_width"], color="green")
ax[1,2].scatter(virginica_df["petal_length"], virginica_df["petal_width"], color="blue")
ax[1,2].set_xlabel("Petal Length (cm)", fontweight = 550)
ax[1,2].set_ylabel("Petal Width (cm)", fontweight = 550)

# Add legend to the figure with appropriate positioning
fig.legend(['setosa','veriscolor','virginica'], loc='upper center', bbox_to_anchor=(0.5,0.92), ncol=3)

# Save to png
plt.savefig('Iris_Scatterplots_2D.png')

##############################
'''Iris_Scatterplots_3D.png'''
##############################

from mpl_toolkits import mplot3d

# Create a new figure and sets of axes, split into 2 rows and 2 columns.
fig, ax = plt.subplots(2,2,subplot_kw={"projection":"3d"})

# Set figure title, size & label
fig.suptitle('Iris Scatterplots (3 Variables)', y=.94, fontweight = 900)
fig.set_figheight(10)
fig.set_figwidth(12)

# First Trio: sepal length vs. sepal width vs. petal length
ax[0,0].scatter(setosa_df["sepal_length"], setosa_df["sepal_width"], setosa_df["petal_length"], color="red")
ax[0,0].scatter(versicolor_df["sepal_length"], versicolor_df["sepal_width"], versicolor_df["petal_length"], color="green")
ax[0,0].scatter(virginica_df["sepal_length"], virginica_df["sepal_width"], virginica_df["petal_length"], color="blue")
ax[0,0].set_xlabel("Sepal Length (cm))", fontweight = 550)
ax[0,0].set_ylabel("Sepal Width (cm)", fontweight = 550)
ax[0,0].set_zlabel("Petal Length (cm)", fontweight = 550)
#ax[0,0].set_box_aspect(None, zoom=0.85) --> This will make the plots fit on the jupyter console.

# Second Trio: sepal length vs. sepal width vs. petal width
ax[0,1].scatter(setosa_df["sepal_length"], setosa_df["sepal_width"], setosa_df["petal_width"], color="red")
ax[0,1].scatter(versicolor_df["sepal_length"], versicolor_df["petal_length"], versicolor_df["petal_width"], color="green")
ax[0,1].scatter(virginica_df["sepal_length"], virginica_df["petal_length"], virginica_df["petal_width"], color="blue")
ax[0,1].set_xlabel("Sepal Length (cm))", fontweight = 550)
ax[0,1].set_ylabel("Sepal Width (cm)", fontweight = 550)
ax[0,1].set_zlabel("Petal Width (cm)", fontweight = 550)
#ax[0,1].set_box_aspect(None, zoom=0.85)

# Third Trio: sepal length vs. petal length vs. petal width
ax[1,0].scatter(setosa_df["sepal_length"], setosa_df["petal_length"], setosa_df["petal_width"], color="red")
ax[1,0].scatter(versicolor_df["sepal_length"], versicolor_df["petal_length"], versicolor_df["petal_width"], color="green")
ax[1,0].scatter(virginica_df["sepal_length"], virginica_df["petal_length"], virginica_df["petal_width"], color="blue")
ax[1,0].set_xlabel("Sepal Length (cm))", fontweight = 550)
ax[1,0].set_ylabel("Petal Length (cm)", fontweight = 550)
ax[1,0].set_zlabel("Petal Width (cm)", fontweight = 550)
#ax[1,0].set_box_aspect(None, zoom=0.85)

# Fourth Trio: sepal width vs. petal length vs. petal width
ax[1,1].scatter(setosa_df["sepal_width"], setosa_df["petal_length"], setosa_df["petal_width"], color="red")
ax[1,1].scatter(versicolor_df["sepal_length"], versicolor_df["petal_length"], versicolor_df["petal_width"], color="green")
ax[1,1].scatter(virginica_df["sepal_length"], virginica_df["petal_length"], virginica_df["petal_width"], color="blue")
ax[1,1].set_xlabel("Sepal Width (cm))", fontweight = 550)
ax[1,1].set_ylabel("Petal Length (cm)", fontweight = 550)
ax[1,1].set_zlabel("Petal Width (cm)", fontweight = 550)
#ax[1,1].set_box_aspect(None, zoom=0.85)

# Add legend to the figure with appropriate positioning
fig.legend(['setosa','veriscolor','virginica'], loc='upper center', bbox_to_anchor=(0.5,0.92), ncol=3)

# Save to png
plt.savefig('Iris_Scatterplots_3D.png')


'''4. CORRELATION'''


corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap='coolwarm')
df.groupby('species').corr(numeric_only=True)


'''5. LOGISTIC REGRESSION'''


# We create a new list called 'target'. Map each species to an integer value and add these values to target.

target = []

for i in y['species']:
    if i == "setosa":
        target.append(0)
    elif i == 'versicolor':
        target.append(1)
    else:
        target.append(2)

# Add additional 'target' column to y DataFrame, and set values equal to 'target' list
y['target'] = target

# Split the data into test and train using the sklearn library function imported
X_train, X_test, y_train, y_test = train_test_split(X, y['target'], test_size=0.3, random_state=0)


# Logistic Regression Implementation
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

predictions = log_reg.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, predictions, labels=[0, 1, 2]),index=[0, 1, 2], columns=[0, 1, 2])