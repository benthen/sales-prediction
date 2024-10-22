import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
color_pal = sns.color_palette()
le = LabelEncoder()

# # grid search cv param
param_grid = {
    'learning_rate': [0.05],
    'max_depth': [7],
    'n_estimators': [300],
    'subsample': [0.8], 
    'colsample_bytree': [1.0],
    'lambda': [1],
}

# xgb param
xgb_params = {
    'random_state': 42,
    'objective' :'reg:squarederror'
}

print("Processing the data...")
# Read the data into a pandas DataFrame
df = pd.read_csv('CodeChallenge_Dataset_2021-2023_Set 1.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'].str.split().str[0])

# Group by 'Date' and 'ProductId', and sum the 'Amount' column
df = df.groupby(['Date', 'ProductId'])['Amount'].sum().reset_index()
df['ProductId'] = le.fit_transform(df['ProductId'])
df = df.set_index('Date')

# df['Amount'].plot(style='.',
#         figsize=(15, 15),
#         color=color_pal[0],
#         title='Sales Prediction by Date and Product ID')
# plt.show()

# remove the negative value which is the outlier
df = df[df['Amount'] >= 0]

# Outlier detection using IQR
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]

# df['Amount'].plot(style='.',
#         figsize=(20, 15),
#         color=color_pal[0],
#         title='Sales Prediction by Date and Product ID')
# plt.show()

# create the features
df['day'] = df.index.day
df['month'] = df.index.month
df['weekday'] = df.index.weekday
df['dayofweek'] = df.index.dayofweek
df['quarter'] = df.index.quarter
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofyear'] = df.index.dayofyear
df['dayofmonth'] = df.index.day
df['weekofyear'] = df.index.isocalendar().week
df['is_month_end'] = df.index.is_month_end.astype(int)
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['weekday_month_interaction'] = df['weekday'] * df['month']

# feature engineering
df['Amount_lag_1'] = df['Amount'].shift(1)  # Sales from the previous day
df['Amount_lag_7'] = df['Amount'].shift(7)  # Sales from 7 days ago
df['Amount_lag_30'] = df['Amount'].shift(30)  # Sales from previous month

# rolling windows
df['Amount_rolling_mean_3'] = df['Amount'].rolling(window=3).mean()  # 3-day rolling mean
df['Amount_rolling_mean_7'] = df['Amount'].rolling(window=7).mean()  # 7-day rolling mean
df['Amount_rolling_mean_30'] = df['Amount'].rolling(window=30).mean()  # 30-day rolling mean

# exponential moving averages
df['ema_3'] = df['Amount'].ewm(span=3).mean()
df['ema_7'] = df['Amount'].ewm(span=7).mean()
df['ema_30'] = df['Amount'].ewm(span=30).mean()

# Cumulative sales trend
df['cumulative_sales'] = df['Amount'].cumsum()

# remove the data with incomplete value
df.fillna(0, inplace=True)

# get all the columns from the dataframe and remove the column Amount
FEATURES = [col for col in df.columns if col != 'Amount']
TARGET = 'Amount'

print("Training the model...")
# Step 2: Define feature columns and target
X = df[FEATURES]  # Features based on the date
y = df[TARGET]  # Target variable (sales amount)

# split the dataset into training and testing
# 80% of the data is training dataset
# 20% of the data is testing dataset
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define the XGBoost model
# GridSearchCV is used to tune the hyperparameter
xgb_model = xgb.XGBRegressor(
    **xgb_params
)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Use the best model from the grid search to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Set up eval_set for training and validation metrics
eval_set = [(X_train, y_train), (X_test, y_test)]
best_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Retrieve evaluation results
results = best_model.evals_result()

epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

# Plot RMSE
# plt.figure(figsize=(10,5))
# plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE')
# plt.plot(x_axis, results['validation_1']['rmse'], label='Validation RMSE')
# plt.xlabel('Epochs')
# plt.ylabel('RMSE')
# plt.title('XGBoost RMSE Over Epochs After GridSearchCV')
# plt.legend()
# plt.show()

print(f"Best Parameters: {grid_search.best_params_}")
print(f"RMSE: {rmse}")