# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 13:38:34 2017

@author: blenderherad
"""
#############################################
#              import packages
#############################################
import pandas as pd
import seaborn as sns
from sklearn import svm

############################################
#               Cleaning data
############################################

# import data set
df = pd.read_csv('train.csv')
testset  = pd.read_csv('test.csv')


# first remove name, cabins and tickets
df.drop('Name',        axis=1, inplace=True)
df.drop('Cabin',       axis=1, inplace=True)
df.drop('Ticket',      axis=1, inplace=True)


# want to fill data for age based on means of sex and Pclass.
# e.g. 1st class females and 3rd class females have different mean ages
# compute means with groupby, round to nearest integer age since that's what is given in dataset
mean_ages = df.groupby(['Sex','Pclass'])['Age'].transform('mean')

# fill missing values
df.loc[:,'Age'] = df['Age'].fillna(mean_ages)

# now replace sex M/F with 0/1 numerical value
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# repalce departure point with numerical index
# 0 = Cherbourg, 1 = Queenstown, S = Southampton
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# most people embarked from Chersbourg, so fill NaN's with 0
df['Embarked'] = df['Embarked'].fillna(0)

# now do a seaborn pairplot
#sns.pairplot(df,hue='Survived')

############################################
#               Modelling
############################################
# time to train a RBF kernel nonlinear SVM
clf = svm.SVC(kernel='rbf',verbose=True)

# do not use passenger ID as attribute
clf.fit(df.drop(['Survived','PassengerId'], axis=1).values, df['Survived'].values)
print clf.get_params()

#print df['Survived'].values

# trim useless values from test set
testset.drop('Name',        axis=1, inplace=True)
testset.drop('Cabin',       axis=1, inplace=True)
testset.drop('Ticket',      axis=1, inplace=True)

# similar imputation of missing data 
# and numericalization as perfomed on training set
mean_ages = testset.groupby(['Sex','Pclass'])['Age'].transform('mean')

# fill missing values
testset.loc[:,'Age'] = testset['Age'].fillna(mean_ages)

# now replace sex M/F with 0/1 numerical value
testset['Sex'] = testset['Sex'].map({'female': 0, 'male': 1})

# repalce departure point with numerical index
# 0 = Cherbourg, 1 = Queenstown, S = Southampton
testset['Embarked'] = testset['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# most people embarked from Chersbourg, so fill NaN's with 0
testset['Embarked'] = testset['Embarked'].fillna(0)

# we have one more missing fare, simply fill with mean
testset.fillna(testset.mean(), inplace=True)

#print np.where(testset.isnull())[0]

predictions = clf.predict(testset.drop(['PassengerId'],axis=1).values)

output = pd.DataFrame({'PassengerId':testset['PassengerId'], 'Survived':predictions})
output.to_csv('rbf_svm.csv', index=False)
