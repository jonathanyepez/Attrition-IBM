# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:46:13 2021

@author: Jonathan A. Yepez M. 
"""
#File to process variables in EDA

#Importing relevant libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from functools import partial

#Defining the function for Categorical variables
def categorical_eda(df, VarOfInt):
    """Given dataframe, generate EDA of categorical data"""
    print("To check: Unique count of non-numeric data")
    print(df.select_dtypes(include=['category']).nunique())
    # Plot count distribution of categorical data
    for col in df.select_dtypes(include='category').columns:
        if col!= 'Attrition':
            fig = sns.catplot(x=col, kind="count", hue=VarOfInt, data=df)
            fig.set_xticklabels(rotation=45)
            plt.show()
        
#Defining the function for Numerical variables


#Feature selection for future analysis
def bestFeature(df, VarOfInt):
    X = df.iloc[:,1:]
    Y = df[VarOfInt] #target column -> i.e. Attrition
    #apply SelectKBest class to extract top 10 best features
    discrete_feat_idx = [2, 4, 7, 9, 13, 15, 19] # an array with indices of discrete features
    score_func = partial(mutual_info_classif, discrete_features=discrete_feat_idx)
    print(score_func)
    bestfeatures = SelectKBest(score_func, k=10)
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features


"""
# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print(model.feature_importances_)
"""