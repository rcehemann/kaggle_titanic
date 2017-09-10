# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 13:38:34 2017

@author: blenderherad
"""
#############################################
#              import packages
#############################################
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, cross_validation, preprocessing
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor



# pre-define classifiers for switching
CLASSIFIERS = {'SVM': svm.SVC(kernel='rbf',verbose=False, tol=1e-8),
               'RF': RandomForestClassifier(max_depth=8, random_state=1),
               'NN': MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=(12,4)),
               'KM': KMeans(n_clusters=2, init='k-means++', n_init=2, max_iter=300)}
CLASSIFIER  = 'RF'
FILENAME    = {'SVM': 'rbf_svm.csv', 'RF': 'ran_for.csv', 'NN': 'neuralnet.csv', 'KM': 'kmeans.csv'}

# SET CLASSIFIER

# options for run
# how to impute ages?
MEAN_IMPUTE = False
# set imputer class
AGE_REGRESSORS = {'KNN': KNeighborsRegressor(), 'LR': LinearRegression()}
AGE_REGRESSOR  =  'KNN'

# hash or classify names?
HASH_NAMES   = False

############################################
#               Cleaning data
############################################

# import data set
df = pd.read_csv('train.csv')
testset  = pd.read_csv('test.csv')
# first remove cabins and tickets
#df.drop('Name',        axis=1, inplace=True)
df.drop('Cabin',       axis=1, inplace=True)
df.drop('Ticket',      axis=1, inplace=True)
#df.drop('PassengerId', axis=1, inplace=True)

# we will drop all names but the last and apply a hash function to them, producing 
# two new features: a hash for last name and a hash for maiden name.

if HASH_NAMES:

    name_hash = [hash(name.split(',')[0]) for name in df.Name.values]
    maid_hash = [hash(name.split('(')[-1].split()[-1].split(')')[0]) if 'Mrs.' in name else hash(name.split(',')[0]) for name in df.Name.values]

    df['Name'] = pd.Series(name_hash, index=df.index)
    df['Maid'] = pd.Series(maid_hash, index=df.index)
    
else:
    
    le_name = preprocessing.LabelEncoder()
    le_maid = preprocessing.LabelEncoder()
    name_lab = preprocessing.LabelEncoder.fit_transform(le_name, 
                                                        [name.split(',')[0] for name in df.Name.values])
    maid_lab = preprocessing.LabelEncoder.fit_transform(le_maid,
                                                        [name.split('(')[-1].split()[-1].split(')')[0] 
                                                        if 'Mrs.' in name else 
                                                        name.split(',')[0] for name in df.Name.values])
    df['Name'] = pd.Series(name_lab, index=df.index)
    df['Maid'] = pd.Series(maid_lab, index=df.index)
    
#hashes    = pd.DataFrame({'NameHash': name_hash, 'MaidHash': maid_hash})


#print pd.concat([df,hashes])
#print df
# want to fill data for age based on means of sex and Pclass.
# e.g. 1st class females and 3rd class females have different mean ages
# compute means with groupby, round to nearest integer age since that's what is given in dataset

# now replace sex M/F with 0/1 numerical value
#df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
le_sex = preprocessing.LabelEncoder()
df['Sex'] = preprocessing.LabelEncoder.fit_transform(le_sex, df['Sex'])
# repalce departure point with numerical index
# 1 = Cherbourg, 2 = Queenstown, 3 = Southampton
le_emb = preprocessing.LabelEncoder()
df['Embarked'] = preprocessing.LabelEncoder.fit_transform(le_sex, df['Embarked'])

if MEAN_IMPUTE:
    mean_ages = df.groupby(['Sex','Pclass'])['Age'].transform('mean')
    df.loc[:,'Age'] = df['Age'].fillna(mean_ages)
    print pd.Series(mean_ages)
    print df
else:
    # perform decision tree prediction of age based on sex, class and fare
    imp_df      = df[['Name','Maid','Sex','Pclass','Age']].dropna(how='any')    
    imputer     = AGE_REGRESSORS[AGE_REGRESSOR].fit(imp_df[['Name','Maid','Sex','Pclass']].values, imp_df['Age'].values)
    imp_ages    = pd.Series(imputer.predict(df[['Name','Maid','Sex','Pclass']].values.tolist()))
#    print imputer.predict(df.groupby(['NameHash','MaidHash','Sex','Pclass']).values)
 #   imp_ages    = df.groupby(['NameHash','MaidHash','Sex','Pclass'])['Age'].transform(imputer.predict(df[['NameHash','MaidHash','Sex','Pclass']].values.tolist()))
#    print imp_ages
    df.loc[:,'Age'] = df['Age'].fillna(imp_ages)      

# fill missing values






# most people embarked from Chersbourg, so fill NaN's with 0
df['Embarked'] = df['Embarked'].fillna(0)

# normalize data; first remove survived

Y = df['Survived'].values
#print Y
df.drop('Survived', axis=1, inplace=True)
df.drop('PassengerId', axis=1, inplace=True)
df = (df - df.mean())/(df.max()-df.min())
#print df

#sns.pairplot(df, hue='Survived')

# now do a seaborn pairplot
#sns.pairplot(df,hue='Survived')

############################################
#               Modelling
############################################
# train a random forest
clf = CLASSIFIERS[CLASSIFIER]
X = df.values
#Y = Y.values

# perform cross-validation testing to estimate score before submission
n_cv = 100
CV10_score = 0
for i in range(n_cv):

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.1)
    CV10_score += clf.fit(X_train, Y_train).score(X_test, Y_test)

print "Ave. cross-validation score:", CV10_score/n_cv

# reset classifier and fit to whole data set
clf = CLASSIFIERS[CLASSIFIER]
clf.fit(X, Y)

############################################
#               Testing
############################################

#print df['Survived'].values

# trim useless values from test set
#testset.drop('Name',        axis=1, inplace=True)
testset.drop('Cabin',       axis=1, inplace=True)
testset.drop('Ticket',      axis=1, inplace=True)

# similar imputation of missing data 
# and numericalization as perfomed on training set

if HASH_NAMES:
    
    name_hash = [hash(name.split(',')[0]) for name in testset.Name.values]
    maid_hash = [hash(name.split('(')[-1].split()[-1].split(')')[0]) if 'Mrs.' in name else hash(name.split(',')[0]) for name in testset.Name.values]

    testset['Name'] = pd.Series(name_hash, index=testset.index)
    testset['Maid'] = pd.Series(maid_hash, index=testset.index)
    
else:
        
    le_name = preprocessing.LabelEncoder()
    le_maid = preprocessing.LabelEncoder()
    name_lab = preprocessing.LabelEncoder.fit_transform(le_name, 
                                                        [name.split(',')[0] for name in testset.Name.values])
    maid_lab = preprocessing.LabelEncoder.fit_transform(le_maid,
                                                        [name.split('(')[-1].split()[-1].split(')')[0] 
                                                        if 'Mrs.' in name else 
                                                        name.split(',')[0] for name in testset.Name.values])
    testset['Name'] = pd.Series(name_lab, index=testset.index)
    testset['Maid'] = pd.Series(maid_lab, index=testset.index)
    
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

# normalize test set with mean and range from training set


#print np.where(testset.isnull())[0]
ids         = testset['PassengerId'].values
testset.drop('PassengerId', axis=1, inplace=True)
testset = (testset - testset.mean())/(testset.max()-testset.min())
predictions = clf.predict(testset.values)

output = pd.DataFrame({'PassengerId':ids, 'Survived':predictions})
#print output
output.to_csv(FILENAME[CLASSIFIER], index=False)
