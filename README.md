# Sales Prediction Model using XGBoost

## Overview of the Sales Prediction Model
The purpose of this model is to predict future sales based on historical data, using XGBoost, a powerful machine learning algorithm. In this project, sales data are aggregated by day and product, and the total sales amount is used as the target variable to train the model. The model is trained using features like the date and the total amount of sales for a product on that date, and it outputs future sales predictions.

## What is XGBoost?
![XBG Training Model](https://miro.medium.com/v2/resize:fit:1000/0*zdmqFZ2nooBRedqC.png)
XGBoost (eXtreme Gradient Boosting) is an optimized and scalable version of gradient boosting, a machine learning algorithm that builds predictive models by combining the strengths of many simpler models (called weak learners), usually decision trees. It is known for its performance and speed, making it one of the most popular algorithms for structured/tabular data.

# Python Setup

## Install package virtualenv
- ```python
    pip install virtualenv

## Create a virtual environment
  - ```python
    python -m venv venv
  
  - Use the virtual environment to install the required packages
      ```python
      .\venv\Scripts\activate

## Install the required packages
  - ```python
    pip install -r requirements.txt

## Project Structure

- **sales_prediction.py**
  - Contains the code for data cleaning, preprocessing, and feature engineering (e.g., aggregating daily sales data for each product).
  - Contains the code to train and evaluate the XGBoost model, including hyperparameter tuning using `GridSearchCV`.
- **CodeChallenge_Dataset_2021-2023_Set 1.csv**
  - The dataset to be used to train the sales prediction model
  - The dataset contains three-year-data which has the daily quantity and amount of each productID
  - The training data is the first two years
  - The testing data is the third year
- **CodeChallenge_Dataset_2021-2023_Set 2.csv**
  - The dataset to be used to evaluate the training model in terms of robustness and scalability
- **requirements.txt**
  - Lists all required dependencies for the project.
- **README.md**:
  - This is the documentation file.

## Features

- **Data Preprocessing**: 
  - Aggregates the daily sales data by summing up the total amount per `ProductID` for each day.
    ```python
    df = df.groupby(['Date', 'ProductId'])['Amount'].sum().reset_index()
  - Tokenise the `ProductID`
    ``` python
    le = LabelEncoder()
    df['ProductId'] = le.fit_transform(df['ProductId'])
  ![Before Processing](/result/Set-1-before_processing.png)
  - Handles missing or incorrect values by removing negative values from the `Amount` column.
    ``` python
    df = df[df['Amount'] >= 0]
  - Remove the outliers by using the Interquartile range(IQR)
    ```python
    # Outlier detection using IQR
    Q1 = df['Amount'].quantile(0.25)
    Q3 = df['Amount'].quantile(0.75)
    IQR = Q3 - Q1

    # Filter out outliers
    df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]
  ![After Processing](/result/Set-1-after_processing.png)

  - Feature Engineering
    - add the features of the dataset in terms of day, month, year, dayofweek, dayofmonth, dayofyear, weekday, quarter, weekofyear, is_month_end, is_weekend, weekday_month_interaction
        ```python
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
    - add the lag feature based on the past sales amount to help predict the future sales
        ```python
        df['Amount_lag_1'] = df['Amount'].shift(1)  # daily lag feature
        df['Amount_lag_7'] = df['Amount'].shift(7)  # weekly lag feature
        df['Amount_lag_30'] = df['Amount'].shift(30)  # monthly lag feature

  - Rolling window
    - calculate statistics such as mean based on specified window size
        ```python
        df['Amount_rolling_mean_3'] = df['Amount'].rolling(window=3).mean()
        df['Amount_rolling_mean_7'] = df['Amount'].rolling(window=7).mean()
        df['Amount_rolling_mean_30'] = df['Amount'].rolling(window=30).mean()

  - Exponential Moving Averages
    - put more weight to the current data compared to old data
    - can react to recent change faster than simple moving average
        ```python
        df['ema_3'] = df['Amount'].ewm(span=3).mean()
        df['ema_7'] = df['Amount'].ewm(span=7).mean()
        df['ema_30'] = df['Amount'].ewm(span=30).mean()
  
- **Sales Prediction Model**: 
  - Utilizes the XGBoost Regressor (`XGBRegressor`) to train a sales prediction model.
  - Uses `GridSearchCV` for hyperparameter tuning to find the best combination of hyperparameters.
  
# Hyperparameter of the training model
### GridSearchCV Hyperparameter
- n_estimators: 
  - The number of trees (iterations) in the ensemble.
- max_depth
  - Maximum depth of each tree, controlling how complex the individual trees are.
- learning_rate
  - Controls the contribution of each tree to the final prediction, affecting convergence speed.
- subsample
  - The fraction of samples used to build each tree, used for regularization.
- colsample_bytree
  - The fraction of features used to build each tree.
- lambda (L2 regularization)
  - Parameters to reduce model complexity and prevent overfitting.

> The best value of the hyperparameter is 
  - colsample_bytree: 1.0
  - learning_rate: 0.05
  - max_depth: 7
  - n_estimators: 300
  - subsample: 0.8
  - lambda: 1

# Result of the model training
![Model Training](/result/model_training.png)

# Data Preprocessing of Dataset 2
![Dataset 2](/result/Set-2-before_processing.png)
![Dataset 2](/result/Set-2-after_processing.png)

# Result of the evaluation
![Evaluation](/result/evaluation.png)

# RMSE Comparison between Training and Evaluation
![RMSE](/result/rmse.png)
