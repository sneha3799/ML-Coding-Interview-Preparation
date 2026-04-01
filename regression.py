from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print(df.describe())
print(df.isna().sum())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape, y_train.shape)

# model training

# lr = LinearRegression()
# lr.fit(X_train, y_train)
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# prediction
y_pred = rf.predict(X_test)

# evaluation
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("R2:", r2)