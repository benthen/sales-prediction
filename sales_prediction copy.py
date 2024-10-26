import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
plt.style.use('fivethirtyeight')
le = LabelEncoder()
color_pal = sns.color_palette()

training_dataset = 'CodeChallenge_Dataset_2021-2023_Set 1.csv'
evaluation_dataset = 'CodeChallenge_Dataset_2021-2023_Set 2.csv'

# this function is used to preprocess the dataset
# remove negative value and remove the outliers
def preprocess_data(dataset, name):
    # Read the data into a pandas DataFrame
    df = pd.read_csv(dataset)

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'].str.split().str[0])
    
    # multiply the 'Quantity' and 'Amount' column
    df['Amount'] = df['Quantity'] * df['Amount']

    # Group by 'Date' and 'ProductId', and sum the 'Amount' column
    df = df.groupby(['Date', 'ProductId'])['Amount'].sum().reset_index()
    df['ProductId'] = le.fit_transform(df['ProductId'])
    df = df.set_index('Date')

    # plot the dataset that has been preprocessed
    df['Amount'].plot(style='.',
            figsize=(15, 15),
            color=color_pal[0],
            title='Sales Prediction by Date and Product ID')
    plt.savefig(f'D:/programming/sales prediction/result/{name}-before_processing.png')

    # remove the negative value which is the outlier
    df = df[df['Amount'] >= 0]

    # Outlier detection using IQR
    Q1 = df['Amount'].quantile(0.25)
    Q3 = df['Amount'].quantile(0.75)
    IQR = Q3 - Q1

    # Filter out outliers
    df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]

    # plot the dataset that has been preprocessed
    df['Amount'].plot(style='.',
            figsize=(20, 15),
            color=color_pal[0],
            title='Sales Prediction by Date and Product ID')
    plt.savefig(f'D:/programming/sales prediction/result/{name}-after_processing.png')

    return df

# this function is used for feature engineering and moving averages
# helps capture the seasonality and trends of the sales pattern
def create_feature(df):
    
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
    df['Amount_lag_14'] = df['Amount'].shift(14)  # Sales from 14 days ago
    df['Amount_lag_30'] = df['Amount'].shift(30)  # Sales from previous month

    # rolling windows
    df['Amount_rolling_mean_3'] = df['Amount'].rolling(window=3).mean()  # 3-day rolling mean
    df['Amount_rolling_mean_7'] = df['Amount'].rolling(window=7).mean()  # 7-day rolling mean
    df['Amount_rolling_mean_14'] = df['Amount'].rolling(window=14).mean()  # 14-day rolling mean
    df['Amount_rolling_mean_30'] = df['Amount'].rolling(window=30).mean()  # 30-day rolling mean

    # exponential moving averages
    df['ema_3'] = df['Amount'].ewm(span=3).mean()
    df['ema_7'] = df['Amount'].ewm(span=7).mean()
    df['ema_14'] = df['Amount'].ewm(span=14).mean()
    df['ema_30'] = df['Amount'].ewm(span=30).mean()

    # Cumulative sales trend
    df['cumulative_sales'] = df['Amount'].cumsum()

    # remove the data with incomplete value
    df.fillna(0, inplace=True)

    return df

# this function is used to split the dataset into training and testing data
def split_data(df):
    
    # split the data into training and testing dataset
    # the training dataset will cover the year 2021 and 2022
    # the testing dataset will cover the year 2023
    train = df.loc[df.index < '01-01-2023']
    test = df.loc[df.index >= '01-01-2023']
    
    # assign all the columns to FEATURES except for 'Amount'
    FEATURES = [col for col in df.columns if col != 'Amount']
    TARGET = 'Amount'
    
    # splitting the training and testing dataset into X and y
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    return X_train, X_test, y_train, y_test, test

# this function is used to train a sales prediction model
# take the argument of @training_dataset which is the sales dataset 1
def train_model(training_dataset):
    
    # preprocess the training dataset
    processed_data = create_feature(preprocess_data(training_dataset, 'Set-1'))
    
    # split the preprocessed dataset into training and testing data
    X_train, X_test, y_train, y_test, test = split_data(processed_data)
    
    # xgb hyperparameter
    # tune the hyperparameter of the training model
    xgb_params = {
        'random_state': 42,
        'base_score': 0.5, 
        'booster': 'gbtree',
        'learning_rate': 0.05,
        'max_depth': 7,
        'n_estimators': 300,
        'subsample': 0.8, 
        'colsample_bytree': 1.0,
        'lambda': 1.0,
        'early_stopping_rounds': 50,
        'objective': 'reg:linear'
    }

    # define the XGB model
    reg = xgb.XGBRegressor(**xgb_params)
    
    # train the model by fitting the training dataset
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)
    
    # fit the testing dataset into the training model for prediction
    test['prediction'] = reg.predict(X_test)

    # plot the graph of the prediction result and save to directory
    ax = test[['Amount']].plot(figsize=(15, 5))
    test['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Data and Prediction')
    plt.savefig('D:/programming/sales prediction/result/model_training.png')

    # calculate the root mean squared error of the training model
    training_rmse = np.sqrt(mean_squared_error(test['Amount'], test['prediction']))
    print(f'RMSE Score of Training Model: {training_rmse:0.2f}')
    
    # return the trained model for evaluation
    return reg, training_rmse

# this function is used to evaluate the model
# take the argument of @training_dataset which is the sales dataset 2
# the trained model will be used to predict the sales of the dataset 2
def evaluate_model(training_dataset, evaluation_dataset):
    
    # get the trained model from the training process
    reg_model, training_rmse = train_model(training_dataset)

    # preprocess the evaluation dataset
    processed_dataset = create_feature(preprocess_data(evaluation_dataset, 'Set-2'))
    
    # assign all the columns to FEATURES except for 'Amount'
    FEATURES = [col for col in processed_dataset.columns if col != 'Amount']
    TARGET = 'Amount'

    X = processed_dataset[FEATURES]  # Features based on the date
    y = processed_dataset[TARGET]  # Target variable (sales amount)


    # use the trained model to predict the evaluation dataset
    y_pred = reg_model.predict(X)
    y_pred = pd.DataFrame(y_pred, columns=['prediction'])
    processed_dataset['prediction'] = y_pred.values
    
    # plot the graph of the prediction result and save to directory
    ax = processed_dataset[['Amount']].plot(figsize=(30, 15))
    processed_dataset['prediction'].plot(ax=ax, style='--')
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Data and Prediction')
    plt.savefig('D:/programming/sales prediction/result/evaluation.png')

    # calculate the root mean squared error of the prediction of the new dataset
    evaluation_rmse = np.sqrt(mean_squared_error(processed_dataset['Amount'], processed_dataset['prediction']))
    print(f'RMSE Score of Evaluation: {evaluation_rmse:0.2f}')
    
    # Plotting the bar chart of the training rmse and evaluation rmse
    plt.figure(figsize=(6, 4))
    plt.bar(['Training RMSE', 'Evaluation RMSE'], [training_rmse, evaluation_rmse], color=['blue', 'green'])
    plt.title('RMSE of Training and Evaluation')
    plt.savefig('D:/programming/sales prediction/result/rmse.png')

evaluate_model(training_dataset, evaluation_dataset)

