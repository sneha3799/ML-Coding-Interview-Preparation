# Case Study 1: Credit Card Fraud Detection
# CIBC processes 250 million transactions per month. You're given a dataset with fields: amount, merchant_category, 
# hour_of_day, location, is_international, card_type, customer_avg_txn, days_since_last_txn, customer_age, and label 
# is_fraud (1.5% fraud rate). Build a fraud detection pipeline that handles the class imbalance and optimizes for 
# high recall.

# import the libraries
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

# load data
df = pd.read_csv("fraud_data.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

# feature engg
# amount deviation
df["amount_ratio"] = df["amount"] / df["customer_avg_txn"] # risky if high
# time deviation
df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h<6 or h>22 else 0)
# international and high value
df["international"] = (df["is_international"]==1) & (df["amount"] > 5000)
# gap
df["gap"] = (df["days_since_last_txn"] > 30).astype(int)
# high value
df["high_value"] = df["amount"] > 5000

# preprocess
X = df.drop(columns=["is_fraud"]).values
y = df.loc[:, "is_fraud"].values

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

# train model
# pipeline
pipeline = Pipeline([
    ("classifier", LogisticRegression())
])
pipeline.fit(X_sm, y_sm)

# predict
y_pred = pipeline.predict(X_test)

# evaluate
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))