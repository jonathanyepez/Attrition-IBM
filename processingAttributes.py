# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:46:13 2021

@author: Jonathan A. Yepez M.
"""
# File to process variables in EDA

# Importing relevant libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from functools import partial


def categorical_eda(df, VarOfInt):  # function for Categorical variables
    """Given dataframe, generate EDA of categorical data"""
    print("To check: Unique count of non-numeric data")
    print(df.select_dtypes(include=['category']).nunique())
    # Plot count distribution of categorical data
    for col in df.select_dtypes(include='category').columns:
        if col != 'Attrition':
            fig = sns.catplot(x=col, kind="count", hue=VarOfInt, data=df)
            fig.set_xticklabels(rotation=45)
            plt.show()


def numerical_eda(df):  # function for Numerical variables
    # sns.pairplot(data=df)
    # plt.show()
    # plot the correlation matrix of salary, balance and age in data dataframe.
    sns.heatmap(df.corr(), annot=True, cmap='Reds')
    plt.show()

# Feature selection for future analysis


def bestFeature_MutualInfoClassif(df, VarOfInt):
    X = df.iloc[:, 1:]
    y = df[VarOfInt]  # target column -> i.e. Attrition
    score_func = partial(mutual_info_classif)
    bestfeatures = SelectKBest(score_func, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(10, 'Score'))  # print 10 best features


def bestFeature_KBest_chi2(df, VarOfInt):
    X = df.iloc[:, 1:]
    y = df[VarOfInt]
    fs = SelectKBest(score_func=chi2, k='all')
    fit = fs.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(10, 'Score'))  # print 10 best features


def bestFeature_ExtraTreesClassif(df, VarOfInt):
    X = df.iloc[:, 1:]  # independent columns
    y = df[VarOfInt]  # target column i.e price range
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)
    # inbuilt class feature_importances-tree classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()
