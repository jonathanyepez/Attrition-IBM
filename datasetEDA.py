# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:41:41 2021

@author: Jonathan A. Yepez M.
"""

# Script to process data regarding Attrition based on IBM dataset

# Import relevant libraries
import pandas as pd
# import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns #already imported in 'processing'
import missingno  # not 1005 sure if we will use this one
import processingAttributes as processing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Preliminary Data Processing
df = pd.read_csv('EmployeeAttrition.csv')  # read the file from the folder
# print(df.info()) #general information about the dataframe
# print(df.head()) #first 3 entries in the dataframe

# we have only one value in the column EmployeeCount. We can delete it
df.drop('EmployeeCount', inplace=True, axis=1)
df.drop('Over18', inplace=True, axis=1)  # all employees assumed to be over18
df.drop('EmployeeNumber', inplace=True, axis=1)  # get rid of the employee ID
df.drop('StandardHours', inplace=True, axis=1)  # column has only one value: 80
standardHrs = 80

# Specify our categorical variables as 'category'
df['Attrition'] = df['Attrition'].astype('category')
df['BusinessTravel'] = df['BusinessTravel'].astype('category')
df['Department'] = df['Department'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['OverTime'] = df['OverTime'].astype('category')
df['EducationField'] = df['EducationField'].astype('category')
df['JobRole'] = df['JobRole'].astype('category')
df['MaritalStatus'] = df['MaritalStatus'].astype('category')
# copy of categorical features
df_categs = df.select_dtypes(include=['category']).copy()


# use label encoding to change data from categorical to int8 in the dataframe
# Encode categorical vars so that we can use feature selection algorithms
categorical_features = ['BusinessTravel', 'Department', 'Gender', 'OverTime',
                        'EducationField', 'JobRole', 'MaritalStatus']

for f in categorical_features:
    colname = f+'_cat'
    df[colname] = df[f].cat.codes
    df.drop(f, axis=1, inplace=True)

df['Attrition'] = df['Attrition'].cat.codes  # change yes/no to 1/0

del f, colname, categorical_features

tempVar = df['Attrition']
df.drop('Attrition', axis=1, inplace=True)
df.insert(0, 'Attrition', tempVar)  # move target to the first column in the df
del tempVar  # delete the temporary variable

# Checking for null values, empty cells
if df.isnull().any(axis=None):
    print("\nPreview of data with null values:\nxxxxxxxxxxxxx")
    print(df[df.isnull().any(axis=1)].head(3))
    missingno.matrix(df)
    plt.show()

# Checking if there are duplicated entries
if len(df[df.duplicated()]) > 0:
    print("No. of duplicated entries: ", len(df[df.duplicated()]))
    # print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head())
else:
    print("No duplicated entries found")

# Define our variable of interest
TargetVar = 'Attrition'  # the name of the column that we will be focusing on

# Running the code to get plots and preliminar information
processing.categorical_eda(df_categs, TargetVar)
# processing.numerical_eda(df)
"""
# Selecting most relevant features from dataframe to perform further analysis
try:
    print("Select KBest with Mutual Info Classifier:")
    processing.bestFeature_MutualInfoClassif(df, TargetVar)
    print("\nSelect features based on Tree Classifier:")
    processing.bestFeature_ExtraTreesClassif(df, TargetVar)
    print("\nSelect features based on KBest and Chi2:")
    processing.bestFeature_KBest_chi2(df, TargetVar)
except Exception as e:
    print(e)
"""
# Preparing data for training and testing
X = df.iloc[:, 1:]
y = df[TargetVar]  # target column -> i.e. Attrition
# feature selection

# Model Training------------------------------
# fit the model -> Logistic Regression


def trainModel(X, y, test_size=0.33):
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = LogisticRegression(solver='lbfgs', max_iter=5000)
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.2f' % (accuracy*100))


# fit the model -> all features
print("----------\nUsing all features: ")
trainModel(X, y)

# fit the model -> chi2 features
print("----------\nUsing chi2 for selection: ")
X_trans_chi2 = processing.bestFeature_KBest_chi2(df, TargetVar)
trainModel(X_trans_chi2, y)

# fit the model -> mutual information features
print("----------\nUsing mutual information selection: ")
X_trans_mutual = processing.bestFeature_MutualInfoClassif(df, TargetVar)
trainModel(X_trans_mutual, y)
