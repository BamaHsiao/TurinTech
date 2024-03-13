#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:28:23 2024

@author: xiaoyao
"""
#%%
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#%%
# Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=cols)

#%%
#basic understanding
print(dataset.shape)
print(dataset.head(5))
print(dataset.describe())
# only for quantitative
print(dataset.groupby("class").size())
# category
#%%
# data visualization
dataset.plot(kind = "box", subplots = True, layout = (2,2), 
             sharex = False, sharey = False)
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()

#%%
array = dataset.values
x = array[:,0:4]
y = array[:,4]
# supervised ML

X_train, X_validation, Y_train, Y_validation = \
    train_test_split(x, y, test_size=0.20, random_state=1)
#training and validation

#%%
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

print(models)

#%%
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
 
 '''
 shuffle is set to be True to prevent implicit order, 
 for example,
 if it's sorted according to the target variable or collected in a time-dependent manner
 '''
 
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
 
'''
train_test_split is used to initially split the data, 
ensuring that there's a final, untouched validation set to test the model's performance 
after selecting and tuning it with k-fold cross-validation on the training set.
'''

#%%
# Compare Algorithms
# Visualization
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

results_array = np.array(results).T  # Transpose to get algorithms as columns
# Create a DataFrame
results_df = pd.DataFrame(results_array, columns=names)
# Get descriptive statistics
desc_stats = results_df.describe()
print(desc_stats)
# SVM seems to be the most appropriate one

#%%
#make predictions
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#%%
'''
Reference:
    Brownlee, J. (n.d.). Your First Machine Learning Project in Python Step-By-Step. Python Machine Learning. https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
'''