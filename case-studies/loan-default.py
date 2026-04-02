# Case Study 3: Loan Default Prediction
# CIBC's lending team needs to predict which mortgage applicants are likely to default. 
# Dataset fields: income, credit_score, employment_years, debt_to_income_ratio, loan_amount, property_value, 
# num_existing_loans, province, employment_type (salaried/self-employed/contract), age, and label defaulted 
# (4% default rate). The model must be explainable for regulatory compliance. Build the pipeline and show how 
# you'd explain individual predictions.

# import the libraries
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

# load data
df = pd.read_csv("loan_default.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

# feature engg
# already has loans
df["already_loans"] = (df["num_existing_loans"]>2).astype(int)
# low credit score
df["low_credit"] = (df["credit_score"] < 700).astype(int)
# loan to value ratio
df["ltv_ratio"] = (df["loan_amount"] / (df["property_value"]+1))
# loan to income ratio
df["lti_ratio"] = (df["loan_amount"]/ (df["income"]+1))

# preprocess
X = df.drop(columns=["defaulted"]).values
y = df.loc[:, "defaulted"].values

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