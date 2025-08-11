# Notes — Introduction to Machine Learning with Python

## Chapter 1

### Types of Machine Learning
*Supervised*: when we provide both **input** and **output** data.  
Example: images of a tumor as input, with the diagnosis and either the tumor is benign or not, as the output.

*Unsupervised*: when we only provide the **input**.
Example: Data of customer entering a store, and the machine identifying the groups with similar preferences.
---

### How the Machine Understands Data
Thinking of your data as a table:  
- Each **entity** is a row.  
- Each **property** of your entity (like gender or age) is a column.  
- Each row (entity) is called a **sample**.  
- Each column is a **feature**.

---

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

---

## Chapter 2

...

---

## Chapter 3

...

