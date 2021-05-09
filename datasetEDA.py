# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:41:41 2021

@author: Jonathan A. Yepez M. 
"""

#Script to process data regarding Attrition based on IBM dataset

#Import relevant libraries
import pandas as pd
#import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns #already imported in 'processing'
import missingno #not 1005 sure if we will use this one
import processingAttributes as processing

#Preliminary Data Processing
df = pd.read_csv('EmployeeAttrition.csv') #read the file that is in the folder
print(df.info()) #general information about the dataframe
print(df.head()) #first 3 entries in the dataframe

#we have only one value in the column EmployeeCount. We can basically delete it
df.drop('EmployeeCount', inplace=True, axis=1)
df.drop('Over18', inplace=True, axis=1) #all employees are assumed to be over18
df.drop('EmployeeNumber', inplace=True, axis=1) #get rid of the employee ID


#Specify our categorical variables as 'category'
df['Attrition'] = df['Attrition'].astype('category')
df['BusinessTravel'] = df['BusinessTravel'].astype('category')
df['Department'] = df['Department'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['OverTime'] = df['OverTime'].astype('category')
df['EducationField'] = df['EducationField'].astype('category')
df['JobRole'] = df['JobRole'].astype('category')
df['MaritalStatus'] = df['MaritalStatus'].astype('category')


tempVar = df['Attrition']
df.drop('Attrition', axis=1,inplace = True)
df.insert(0, 'Attrition', tempVar) #moving our variable of interest to the first column in the df
del tempVar #delete the temporary variable to prevent cluttering our workspace

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

#Define our variable of interest
VarofInt = 'Attrition' #the name of the column that we will be focusing on

#Running the code to get plots and preliminar information
processing.categorical_eda(df, VarofInt)

#Selecting the most relevant attritubtes from the dataframe to perform further analysis
processing.bestFeature(df, VarofInt)