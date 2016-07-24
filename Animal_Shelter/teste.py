# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import TransformerMixin
import numpy as np


# Load the data
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

# We'll impute missing values using the median for numeric columns and the most common value for string columns.

class DataFrameImputer(TransformerMixin):
    
    def fit(self, X, y=None):
        
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)


def converte(database):
    
    for f in range(0, len(database)):
        temp = str(database[f]).replace('s','').split(' ')
        
        if len(temp) > 1:
            if temp[1]=='year':
                database[f] = str(int(temp[0])*365)
            elif temp[1]=='month':
                database[f] = str(int(temp[0])*30)
            elif temp[1]=='week':
                database[f] = str(int(temp[0])*7)
            else:
                database[f] = '1'
        
    return database
    
train_df['AgeuponOutcome'] = converte(train_df['AgeuponOutcome'])
test_df['AgeuponOutcome'] = converte(test_df['AgeuponOutcome'])

# feature selection
'''    
array =  big_X_imputed.values
X = array[:,0:9]
Y = array[:,9]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
#### pclass, name, age, ticket, fare
'''
feature_columns_to_use = ['AnimalType','SexuponOutcome','AgeuponOutcome','Breed','Color']#['Pclass',  'Age', 'Parch', 'Fare']
nonnumeric_columns = ['AnimalType','SexuponOutcome','Breed','Color']

# Join the features from train and test together before imputing missing values, in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['OutcomeType']

#Predicting multiple classes with "multi:softprob" objective
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, gamma=0.2, objective="multi:softprob").fit(train_X, train_y)
predictions = gbm.predict(test_X)

# Preparing results for submission format
res = []
res.append(feature_columns_to_use)
a=[]
b=[]
c=[]
d=[]
e=[]
for i in range(len(predictions)):
    a.append(1 if predictions[i] == 'Adoption' else 0)
    b.append(1 if predictions[i] == 'Died' else 0)
    c.append(1 if predictions[i] == 'Euthanasia' else 0)
    d.append(1 if predictions[i] == 'Return_to_owner' else 0)
    e.append(1 if predictions[i] == 'Transfer' else 0)


submission = pd.DataFrame({ '.ID': test_df['ID'],
                            'Adoption': a,
                            'Died' : b,
                            'Euthanasia': c,
                            'Return_to_owner': d,
                            'Transfer' : e})
submission.to_csv("submission.csv", index=False)

print("End!")