# Notes — Introduction to Machine Learning with Python

## Chapter 1

### Types of Machine Learning
*Supervised*: when we provide both **input** and **output** data.  
Example: images of a tumor as input, with the diagnosis and either the tumor is benign or
not, as the output.

*Unsupervised*: when we only provide the **input**.  
Example: Data of customer entering a store, and the machine identifying the groups with
similar preferences.

### How the Machine Understands Data
Thinking of your data as a table:  
- Each **entity** is a row.  
- Each **property** of your entity (like gender or age) is a column.  
- Each row (entity) is called a **sample**.  
- Each column is a **feature**.

### What to Keep in Mind While Building an ML Solution
- What question do I have, and can the data collected answer it?  
- Do I have enough data to represent the problem?  
- Can the features of the data enable the right prediction?  
- How will I measure success in my application?

### All the code will use
- import numpy as np
- import matplotlib.pyplot as plt
- import pandas as pd
- import mglearn
- from IPython.display import display

### Machine performance
Usually, it is done by splitting the data in two parts, with one part being used to build the
machine learning model, called **training data**, and the other part is used to assess how
well the model works, called **test data**.

The **train_test_split** function from scikit learn divides and shuffles the dataset for us,
using 75% for the training data part, and 25% for the test data part.

### Looking into the data
It is good to see your data before starting to build a machine learning model, to check if we
find any abnormalities (common in the real world), like differences in measuring units or
maybe too many outliers.

### k-Nearest Neighbors
We give the machine a new example for it to label, so it puts the example in the dataframe
given in the training data, close to the ones with similar measurements, and counts **k**
distances from the place the example is, generalizing by counting the labels in the other
entities, and giving our example the same label.

### Prediction method
We can make a prediction by creating a new test array, and the machine will return to us a
class for the array (features and characteristics we gave to the sample). But we can only be
sure the machine is guessing correctly by giving it the test data, which contains the correct
class, and comparing it to the answer the machine gives.

### Chapter summary
- introduction to machine learning and its applications  
- supervised and unsupervised learning  
- tools used in the book  
- jupyter notebook with all of the practical examples done while reading, which contains:  
- predicting the species of iris of a flower by the measurements using a dataset of
  measurements as data  
- X containing the data (two dimensional array of features) and y containing the correct
  outputs (one dimensional containing class label)  
- split the data into training set to build our model, and test set to see if it works well  
- k-Nearest Neighbors classification that takes the closest neighbors to compare and see in
  which class the new flower fits  
- giving the X_train (training data) and the y_train (training outputs) to teach our model  
- then we give the X_test and compare the response with the y_test to see if it learned  
- machine got a 97% accuracy, giving it certain reliability

### Comments on the chapter
I think I have learned a little about how the machine is trained to get a good accuracy. I
think the next chapter will have info on how to maximize the accuracy of trained models, and
I am excited for that. Of course, I will have to train this process to fix it in my head, but
the general context of how it is done is well learned.

---

## Chapter 2

### Classification in supervised learning
Classification is when we have to predict a class label, from a predefined list of choices,
like the tree species of irises seen in the last chapter. The problem is divided into two
classifications: **binary** and **multiclass**.  

The multiclass classification is the one used in chapter one, because we had three label
options (species of iris). It is used when there are more than two of those.  

The binary classification is when we have only two possible labels, like answering a
true/false question. An example would be a machine that identifies if an email is spam or
not. In this type of classification, one class is considered positive and the other negative.
So, in our example, spam is positive, because the machine is built to look for spam.

### Regression in supervised learning
Regression is when the machine is trying to predict a **real number**, in mathematical terms,
or a **floating point number** in programming terms (a number with a decimal part, even if it
is zero). An example is when the program tries to predict a person's age or income, and we
provide the machine data to train (supervised learning).

### Regression and Classification differences
Basically, one is trying to guess a number, and the other a class. When we don't mind small
changes in the final answer, it is a regression problem. For example, if the machine is to
guess an income of 10,000, and it guesses 9,999, it’s not a big deal.  

In contrast, we do mind in classification problems. If we are trying, for example, to guess
the language of a book, and the machine does not guess the exact language, it is a big
problem.

### Overfitting
It occurs when we try to create too complex a model for little data. Basically, when we give
too specific data to the model, it works well on the test data because we told it exactly
what it needed to get that part right, but when we give it new data, it is not able to
generalize well, because it is too focused on the little details.

### Underfitting
The exact opposite of overfitting. It happens when we make a model that has low
specifications, so it also does not generalize well, because it does not have enough details
to get the right answer.

### Model complexity and Data size
It is important to know that the complexity of the model and the data size are connected in a
way that the more complex the model is, more data is needed, and the more data we have, more
of a complex model we can make without overfitting. BUT just duplicating data points or
collecting too similar data does not work in that case.

### Bunch objects
- are included in scikit learn datasets
- contain info about the dataset as well as the actual data
- example: can be acessed by "bunch.key" instead of the normally used "bunch["key"]"


### KNN neighbors
Is going to be used again, as used in chapter one, is the simplest ML algorithim, 
because building the model consists only of storing the training dataset

Instead of considering only the closest neighbor, it is possible to consider a number "k"
of neighbors, using for example: "(n_neighbors=3)

It can be used not just for binary classification problems, but to datasets with any number of 
classes, in that case, we count how many neighbors belong to each class and predict the most
common class.

---

## Chapter 3

...
