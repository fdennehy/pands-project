# pands-project

## About This Project

This repository contains a Python program (analysis.py) which explores the well-known Fisher's Iris data set using analytical methods learned as part of the Programming and Scripting module of the [HDip in Science in Computing in Data Analytics](https://www.gmit.ie/higher-diploma-in-science-in-computing-in-data-analytics), ATU Galway. In addition to the Python program, the repository contains this README.md file, a Jupyter Notebook (iris.ipynb), an iris_species.png file and a .gitignore file. A brief description of each file is provided under the 'Files Description' section.

## Fisher's Iris Data Set

[Fisher's Iris Data Set](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a set of data collected by [Edgar Anderson](https://en.wikipedia.org/wiki/Edgar_Anderson), and subsequently made famous by [Ronald Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher) "in his 1936 paper _The use of multiple measurements in taxonomic problems_ as an example of linear discriminant analysis". The data set contains 150 records of Iris flowers (50 records for each of the three Iris species recorded) and each record has five attributes:
- _Sepal length_
- _Sepal width_
- _Petal length_
- _Petal width_
- _Species_

![iris_species.png](Iris_species.png)
Source: https://miro.medium.com/v2/resize:fit:1400/format:webp/1*nfK3vGZkTa4GrO7yWpcS-Q.png

Due to its simplicity, standardization and versatility, this data set has become popular for testing machine learning techniques, particularly classifaction techniques such as [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine) and [k-nearest neighbours (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).

## Files Description

1. **analysis.py**\
_Description_: Python program made up of several smaller programs:
- *variable_summary.py*: Outputs a summary of each variable to a single text file.
- *histogram.py*: Saves a histogram of each variable to png files.
- *scatterplot.py*: Outputs a scatter plot of each pair of variables.
- *additional_analysis.py*: Performs additional analysis <TBC>.

2. **iris.ipynb**\
_Description_: Jupyter Notebook containing the following sections:
- _Title_: High level overview.
- _Import_: Summary of the Python packages imported.
- _Load Data & Data Overview_: Initial data load and high level overview of this data.
- _Variables_: Table providing summary of the variables in the dataset and the types of variables that should be used to model these variables, with rationale.
- _Inspect Data_: Deeper dive into the data, with the aim of providing some insights.
- _Visualizations_: Using appropriate plots (bar charts, histograms and scatterplots) to visualize different data.
- _Summary_: Summary of findings

3. **README.md**\
_Description_: This README has been written with [GitHub's documentation on READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) in mind.

4. **.gitignore**\
_Description_: I have used the below github templates to create my .gitignore file:\
    https://github.com/github/gitignore/blob/main/Python.gitignore \
    https://github.com/github/gitignore/blob/main/Global/Windows.gitignore \
    https://github.com/github/gitignore/blob/main/Global/macOS.gitignore \

5. **.iris_species.png**\
_Description_: An image to help visualise the different measurements within the data set.

## Use of this Project

This project may be useful to prospective students of the HDip in Science in Computing in Data Analytics course at ATU Galway, giving an indication of the content of the Programming and Scripting moduule and showcasing what can be achieved within three months of the course. It may also be useful to other Python learners beginning their Data Analytics journey.

## Get Started 

You can jump straight to the notebook using the following clickable link, which was generated using [openincolab.com/](https://openincolab.com/)

This opens the `iris.ipynb` notebook in [Google Colab](https://colab.research.google.com/).

<a target="_blank" href="https://colab.research.google.com/github/fdennehy/pands-project/blob/main/iris.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

To run the files locally, clone the repository and then run the python files locally.

### Cloning the Repository

1. Open your terminal or command prompt. (I use [cmder](https://cmder.app/))
2. Navigate to the directory where you want to clone the repository.
3. Use the following command to clone the repository:
```bash
git clone https://github.com/fdennehy/pands-project
```

### Running Python Files

1. After cloning the repository, navigate into the repository's directory through the command prompt:
```bash
cd repository_name
```
Replace <repository_name> with the name of the directory under which you cloned the repository.

2. Once inside the repository's directory, you can run the .py scripts using the Python interpreter. To to run the analysis.py script, use the following command:
```bash
python analysis.py
```

Now you're ready to explore and use the Python files in the repository! 

## Get Help

Read the comments provided within the Jupyter Notebook and look up official Python documentation for further usage guidance.

## Contribute

Developers are welcome to fork this repo and continue to develop and expand upon it as they wish.

## Author

**Finbar Dennehy**

I'm currently undertaking the [HDip in Science in Computing in Data Analytics](https://www.gmit.ie/higher-diploma-in-science-in-computing-in-data-analytics) on a part time basis at [ATU](https://www.atu.ie/)

I have over ten years' experience in capital markets consultancy and have spent the past few years working on software delivery and customer success. I am undertaking this program to better understand our clients, who are predominantly data scientists and data engineers.

## Acknowledgements

Special thanks to my lecturer on the Programming and Scripting module, Andrew Beatty, from whom I acquired the skills necessary to put this project together.
Now I'm going to stand up, strecth and grab myself a cup of tea!