# https://medium.com/@debspeaks/imbalanced-data-classification-oversampling-and-undersampling-297ba21fbd7c 
# Oversampling — Generate new samples for the class which is under-represented.
# Undersampling — Remove samples from the class which is over-represented.

# As the over sampler creates copies of the minority class, 
# as a result over sampling technique in a way increases the 
# probability for over-fitting.

# The under sampler removes huge amount of rows from the majority 
# class and hence it poses serious threat for under-fitting.

# For majority of the cases, oversampling is preferred more than undersampling. 
# Removing data points is not ideal as it may carry significant piece of information. 

# Instead of creating copies of existing instances of minority class like 
# RandomOverSampler, SMOTE generates new illustrations through interpolation.

# SMOTE only for supervised models like:
# Logistic Regression
# XGBoost
# Random Forest

# SMOTE is useful for supervised fraud classification but not 
# for anomaly detection since anomalies must remain rare.

# Approach 1 (supervised fraud detection)
# Use SMOTE + classifier.

# Approach 2 (anomaly detection baseline)
# Use IsolationForest without SMOTE.

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

# load data
df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.shape)
print(df['Class'].value_counts())
print(df.isna().sum())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# oversampling to handle class imbalance
#creating the instance for SMOTE
smote = SMOTE()

#Resampling with SMOTE
X_smote, y_smote = smote.fit_resample(X, y)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, y_train.shape)

# load model - XGBoost, Isolation Forest
# isf = IsolationForest(n_estimators=100, random_state=42)
xgbc = xgb.XGBClassifier(n_estimators=2)

# train model
xgbc.fit(X_train, y_train)

# make predictions
y_pred = xgbc.predict(X_test)
# y_pred = [1 if x==-1 else 0 for x in y_pred]

# evaluation
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))