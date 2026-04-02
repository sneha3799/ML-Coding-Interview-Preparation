# Data quality issues to clean:

# amount is mixed — some are floats, some are strings like "$79.92", some are negative, some are None
# merchant_category has inconsistent casing ("retail", "Retail", "RETAIL") and typos ("resturant")
# timestamp has 3 different formats plus nulls (ISO, MM/DD/YYYY, DD-Mon-YYYY)
# is_international is a mess — True, "yes", "Yes", 1, None all mixed
# customer_age has impossible values (-5, 0, 150, 999)
# card_type has inconsistent casing
# location has empty strings and nulls
# 50 duplicate rows
# 2% fraud rate (class imbalance)

import pandas as pd

# Step 1: Explore the data
df = pd.read_csv("messy_transactions.csv")
print(df.info())
print(df.describe())
print(df.head(10))
print(df.isnull().sum())
print(df.duplicated().sum())

# Step 2: Drop duplicates and irrelevant column
df = df.drop_duplicates()
# Which columns are identifiers, not features? Drop them.
# Hint: txn_id and customer_id are IDs

# Step 3: Fix amount
def clean_amount(val):
    # handle None
    if val is None or pd.isna(val):
        return None
    
    # handle string with "$"
    if isinstance(val, str):
        val = val.replace("$", "").strip()

    # handle negative values
    val = float(val)
    if val < 0:
        val = abs(val)  # treat negatives as absolute values
    # return float
    return val
df["amount"] = df["amount"].apply(clean_amount)

# Step 4: Standardize categorical columns
# merchant_category: fix casing + typo
df["merchant_category"] = df["merchant_category"].str.lower().str.strip()
df["merchant_category"] = df["merchant_category"].replace({"resturant": "restaurant"})

# card_type: just fix casing
df["card_type"] = df["card_type"].str.lower().str.strip()

# location: fix empty strings to NaN, then strip
df["location"] = df["location"].replace("", None)
df["location"] = df["location"].str.lower().str.strip()

# Step 5: Fix customer_age
def clean_age(val):
    if val is None or pd.isna(val):
        return None
    if val < 18 or val > 100:
        return None
    return val
df["customer_age"] = df["customer_age"].apply(clean_age)

# Step 6: Parse timestamp and engineer features
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek # 0=Monday, 6=Sunday
df = df.drop(columns=["timestamp"])

# Step 7: Feature engineering
# Is this transaction unusually large for this customer?
df["amount_ratio"] = df["amount"] / df["customer_avg_txn"]
# A ratio of 5.0 means "5x their usual spend" — suspicious

# How far from average?
df["amount_deviation"] = df["amount"] - df["customer_avg_txn"]

# Late night transaction?
df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h > 22 else 0)

# Unusually long gap since last transaction?
df["long_gap"] = (df["days_since_last_txn"] > 30).astype(int)

# High value transaction?
df["high_amount"] = (df["amount"] > 1000).astype(int)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # Fit and transform the data
# X_scaled = scaler.fit_transform(X_train)

# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# import numpy as np

# # Sample data
# data = np.array([['Apple'], ['Mango'], ['Orange'], ['Apple']])
# encoder = OneHotEncoder(sparse_output=False)
# onehot = encoder.fit_transform(data)

# print(onehot)
# # Output:
# # [[1. 0. 0.]  <- Apple
# #  [0. 1. 0.]  <- Mango
# #  [0. 0. 1.]  <- Orange
# #  [1. 0. 0.]] <- Apple

# le = LabelEncoder()
# df["merchant_encoded"] = le.fit_transform(df["merchant_category"])


# Encoding — Making Categories into Numbers
# Models only understand numbers. Encoding converts categories to numbers. There are two main approaches:
# LabelEncoder — assigns a number to each category:
# pythonfrom sklearn.preprocessing import LabelEncoder

# # Before: ["retail", "online", "atm", "restaurant"]
# # After:  [2, 1, 0, 3]

# le = LabelEncoder()
# df["merchant_encoded"] = le.fit_transform(df["merchant_category"])

# The problem: The model thinks restaurant (3) > retail (2) > online (1) — it assumes an ordering that doesn't exist. Use LabelEncoder only for ordinal data (low/medium/high) or tree-based models (XGBoost, Random Forest) which split on individual values and don't care about ordering.
# OneHotEncoder — creates a separate column for each category:
# pythonfrom sklearn.preprocessing import OneHotEncoder

# # Before: ["retail", "online", "atm"]
# # After:
# #   retail  online  atm
# #      1       0     0
# #      0       1     0
# #      0       0     1
# ```

# Each category gets its own binary column. No false ordering. Use this for **linear models** (LogisticRegression, SVM) and when you have few categories.

# **When to use which:**
# ```
# LabelEncoder  → tree models (XGBoost, RF), ordinal data
# OneHotEncoder → linear models, <15 categories
# Neither       → 100+ categories (use target encoding or embeddings instead)

# Scaling — Putting Features on the Same Scale
# amount ranges 0–10,000. hour_of_day ranges 0–23. Without scaling, the model thinks amount is more important just because the numbers are bigger.
# StandardScaler — centers around 0, spread of 1:
# from sklearn.preprocessing import StandardScaler

# # Before: [100, 200, 5000, 150]
# # After:  [-0.5, -0.4, 2.1, -0.45]
# # Formula: (value - mean) / std
# Use for: LogisticRegression, SVM, Neural Networks — anything that uses distances or gradients.
# MinMaxScaler — squeezes to 0–1 range:
# from sklearn.preprocessing import MinMaxScaler

# # Before: [100, 200, 5000, 150]
# # After:  [0.0, 0.02, 1.0, 0.01]
# # Formula: (value - min) / (max - min)
# Use for: Neural Networks, when you need bounded values.
# When to skip scaling: Tree-based models (XGBoost, Random Forest, Decision Trees) don't need scaling — they split on thresholds, so the magnitude doesn't matter.

# Imputation — Filling Missing Values
# Missing values crash most models. You need to fill them with something reasonable.

# SimpleImputer strategies:
# from sklearn.impute import SimpleImputer

# # Numeric columns: fill with median (robust to outliers)
# num_imputer = SimpleImputer(strategy="median")

# # Categorical columns: fill with most frequent value
# cat_imputer = SimpleImputer(strategy="most_frequent")

# # Or fill with a constant
# cat_imputer = SimpleImputer(strategy="constant", fill_value="unknown")
# Why median over mean? If amounts are [100, 150, 200, 50000], the mean is 12,612 (skewed by the outlier). The median is 175 — much more representative.



# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from xgboost import XGBClassifier

# numeric_features = ["amount", "customer_avg_txn", "days_since_last_txn", 
#                      "customer_age", "hour_of_day", "day_of_week",
#                      "amount_ratio", "is_night"]

# categorical_features = ["merchant_category", "card_type", "location"]

# # Numeric: impute missing with median, then scale
# numeric_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler()),
# ])

# # Categorical: impute missing with "unknown", then one-hot encode
# categorical_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
#     ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
# ])

# # Combine
# preprocessor = ColumnTransformer([
#     ("num", numeric_pipeline, numeric_features),
#     ("cat", categorical_pipeline, categorical_features),
# ])

# # Full pipeline: preprocess → model
# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("classifier", XGBClassifier(
#         scale_pos_weight=49,  # ratio of non-fraud to fraud (98/2)
#         random_state=42,
#         eval_metric="logloss",
#     )),
# ])