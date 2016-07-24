# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import TransformerMixin
import numpy as np


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

# If you want to perform a feature selection before train, you can descomment this code block below
feature_columns_to_use = ['Pclass',  'Age', 'Parch', 'Fare']
nonnumeric_columns = ['Name', 'Sex', 'SibSp', 'Ticket', 'Cabin', 'Embarked']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)
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

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more details and options
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

# You can experiment with many other options here, using the same .fit() and .predict() methods; see http://scikit-learn.org
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.1, subsample=0.8,
 colsample_bytree=0.8, gamma=0.2).fit(train_X, train_y)
predictions = gbm.predict(test_X)

#Submission style
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)
