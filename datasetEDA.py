# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:41:41 2021

@author: Jonathan A. Yepez M. 
"""

#Script to process data regarding Attrition based on IBM dataset

#Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno #not 1005 sure if we will use this one

#Preliminary Data Processing
df = pd.read_csv('EmployeeAttrition.csv') #read the file that is in the folder
print(df.info()) #general information about the dataframe
print(df.head()) #first 3 entries in the dataframe

#we have only one value in the column EmployeeCount. We can basically delete it
df.drop('EmployeeCount', inplace=True, axis=1)
df.drop('Over18', inplace=True, axis=1) #all employees are assumed to be over18
df.set_index('EmployeeNumber') #use the employee ID column as index

#Specify our categorical variables as 'category'
df['Attrition'] = df['Attrition'].astype('category')
df['BusinessTravel'] = df['BusinessTravel'].astype('category')
df['Department'] = df['Department'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['OverTime'] = df['OverTime'].astype('category')

#Checking for null values, empty cells
if df.isnull().any(axis=None):
    print("\nPreview of data with null values:\nxxxxxxxxxxxxx")
    print(df[df.isnull().any(axis=1)].head(3))
    missingno.matrix(df)
    plt.show()
    
#Checking if there are duplicated entries
if len(df[df.duplicated()]) > 0:
    print("No. of duplicated entries: ", len(df[df.duplicated()]))
    print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head())
else:
    print("No duplicated entries found")

def categorical_eda(df):
    """Given dataframe, generate EDA of categorical data"""
    print("To check: Unique count of non-numeric data")
    print(df.select_dtypes(include=['category']).nunique())
    # Plot count distribution of categorical data
    for col in df.select_dtypes(include='category').columns:
        fig = sns.catplot(x=col, kind="count", data=df)
        fig.set_xticklabels(rotation=90)
        plt.show()

categorical_eda(df)

sns.catplot(x='Department', kind='count', hue='Attrition', data=df)