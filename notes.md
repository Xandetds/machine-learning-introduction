# Notes — Introduction to Machine Learning with Python

## Chapter 1

### Types of Machine Learning
*Supervised*: when we provide both **input** and **output** data.  
Example: images of a tumor as input, with the diagnosis and either the tumor is benign or not, as the output.

*Unsupervised*: when we only provide the **input**.
Example: Data of customer entering a store, and the machine identifying the groups with similar preferences.

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
Usually, is done by splitting the data in two parts, with one part being
used to build the machine learning model, called **training data**, and 
the other part is used to assess how well the model works, called **test data**

The **train_test_split** function from scikit learn divides ans shuffles the
dataset for us, using 75% for the training data part, and 25% for the test data part

### Looking into the data
It is good to see your data before starting to build a machine learning model, to see
if we find any abnormalities (happens often in the real world), like differences in 
measuring units or maybe too much outliers

### k-Nearest Neighbors
We give the machine a new example for it to label, so it puts the example in the dataframe
given in the training data, close to the ones with similar measurements, and counts **k**
numbers of distance from the place the example is, and generalizes by counting the labels in
the other entities, giving our example the same label

### Prediction method
We can make a prediction by creating a new test array, and the machine will return to us a class
for the array (features and characteristics we gave to the sample), but we can only be sure the machine
is guessing correctly, by giving it the test data, that i mentioned earlier here, on which we have the correct
class for, and can compare to the answer the machine will be giving to us.


### Chapter summary
- introduction to machine learning and its applications
- supervised and unsupervised learning
- tools used in the book 
- jupyter notebook with all of the practical examples i did while reading, which contains:
- predicting the species of iris of a flower by the measurements using a dataset of measurements as data
- X containing the data (two dimensional array of features) and y containing the correct outputs (one dimensional containing class label)
- split the data into training set to build our model, and test set to see if it works well
- k-Nearest Neighbors classifications thas takes the closest neighbors to compare and see in which class the new flower fits
- giving the X_train (training data) and the y_train (training outputs) to teach our model
- then we give the X_test and compare the response with the y_test to see if it learned
- i found out the machine got a 97% accuracy, and giving it certain confiability

### Comments on the chapter
I really think i have learned a little about how the machine is trained to get a good accuracy, i think the next chapter will have
info on how to maximize the accuracy of trained models, and i am excited for that, of course i am going to have to train this type of
process, to fixate it good in my head, but the general context of how it is done have been pretty well learned

---

## Chapter 2

### Classification in supervised learning
Classification, is when we have to predict a class label, from a predefined list of
choices, like the tree species of irises seen in the last chapter. The problem is divided in two classifications, the **binary** and the **multiclass**.
The multiclass classification is the one used in chapter one, because we had three options of label (species of iris), so, it is used when there 
are more than two of those.
The binary classification is when we have only two possible labels, so it is like the machine is answering a true/false question, an example would be
a machine that identifies either the mail you recieved is a spam email or not. In this type of classification, it is often spoken of one class being 
the positive one, and the other, the negative, so, in our example, the mail being a spam, the object is positive, because the machine was built to look 
for a spam.

### Regression in supervised learning
Regression is when the machine is trying to predict a **real number**, in mathematical terms, or a **floating point number** in programming terms, it is 
basically a decimal part in it, even if it is zero. An example is when the program tries to predict a persons age, or income, and we provide the machine 
data to train (supervised learning). 

### Regression and Classification diferences
Basically, one is trying to guess a number, and the other a class. When we dont mind that much in changes in the final answer, we know is a regression problem,
because, even if the machine is to guess an income of 10,000, and it guesses 9,999 we dont mind that much. In contrast of that, we do mind when the machine 
makes this type of error in classification, because if we are trying for example, to guess a language on a book, if the machine does not guess the exact language,
it is a big problem.

### Overfitting
It occours when we try to create too complex of a model for little data, basically is when we give too especific data to the model, and it works well on the test
data, because we did especify what it needed to know to get that part right, but when we give the model new data, it is not able to generalize well, because is too
focused on the little details

### Underfitting 
The exact opposite of overfitting, it happens when we make a model that has low especifications, so it also does not generalize well, because is does not have enough details to get the right answer.


---

## Chapter 3

...

