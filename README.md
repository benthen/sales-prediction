# Sales Prediction Model using XGBoost

## Overview of the Sales Prediction Model
The purpose of this model is to predict future sales based on historical data, using XGBoost, a powerful machine learning algorithm. In this project, sales data are aggregated by day and product, and the total sales amount is used as the target variable to train the model. The model is trained using features like the date and the total amount of sales for a product on that date, and it outputs future sales predictions.

## What is XGBoost?
![XBG Training Model](https://miro.medium.com/v2/resize:fit:1000/0*zdmqFZ2nooBRedqC.png)
XGBoost (eXtreme Gradient Boosting) is an optimized and scalable version of gradient boosting, a machine learning algorithm that builds predictive models by combining the strengths of many simpler models (called weak learners), usually decision trees. It is known for its performance and speed, making it one of the most popular algorithms for structured/tabular data.

### Parameter Tuning
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
- lambda (L2 regularization) and alpha (L1 regularization)
  - Parameters to reduce model complexity and prevent overfitting.

## Project Structure

- **data_processing.py**: Contains the code for data cleaning, preprocessing, and feature engineering (e.g., aggregating daily sales data for each product).
- **model_training.py**: Contains the code to train and evaluate the XGBoost model, including hyperparameter tuning using `GridSearchCV`.
- **requirements.txt**: Lists all required dependencies for the project.
- **README.md**: This documentation file.
- **model_training.ipynb**: This documentation file.

## Features

- **Data Preprocessing**: 
  - Aggregates the daily sales data by summing up the total amount per `ProductID` for each day.
  - Handles missing or incorrect values by removing negative values from the `Amount` column.
  
- **Sales Prediction Model**: 
  - Utilizes the XGBoost Regressor (`XGBRegressor`) to train a sales prediction model.
  - Uses `GridSearchCV` for hyperparameter tuning to find the best combination of hyperparameters.
  
## Data Preparation

Ensure the data is cleaned and aggregated properly before feeding it into the model. The key steps for data preparation include:

1. **Summing Up the Total Sales Per Day**: 
   The `Amount` for each product per day is summed to provide the target for training the model.

   ```python
   daily_sales = data.groupby(['Date', 'ProductId']).agg({'Amount': 'sum'}).reset_index()
