import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from model import load_model, load_data

data = load_data('Ecommerce_Customers')
model = load_model('lr_model.pkl')

def calculate_residuals(data: pd.DataFrame, model: BaseEstimator) -> np.ndarray:
    """Calculate residuals for the model predictions.
    
    Args:
        data (pd.DataFrame): The input data containing features and target.
        model (BaseEstimator): The trained model used for predictions.
    
    Returns:
        np.ndarray: The residuals of the predictions.
    """
    # Assuming the target variable is 'Yearly Amount Spent'
    X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y_true = data['Yearly Amount Spent']
    y_pred = model.predict(X)
    residuals = y_true - y_pred
    return residuals

def calculate_metrics(data: pd.DataFrame, model: BaseEstimator) -> dict:
    """Calculate performance metrics for the model.
    
    Args:
        data (pd.DataFrame): The input data containing features and target.
        model (BaseEstimator): The trained model used for predictions.
    
    Returns:
        dict: A dictionary containing RMSE, MSE, and R² metrics.
    """
    X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y_true = data['Yearly Amount Spent']
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MSE': mse,
        'R²': r2
    }

def get_baseline_metrics(data: pd.DataFrame) -> dict:
    """Calculate baseline metrics using the mean of the training set.
    
    Args:
        data (pd.DataFrame): The input data containing features and target.
    
    Returns:
        dict: A dictionary containing baseline R², MAE, and MSE metrics.
    """
    X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = data['Yearly Amount Spent']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate the mean of the training set
    mean_price = y_train.mean()
    
    # Predict the mean price for all test instances
    y_pred_baseline = np.full(y_test.shape, mean_price)
    
    # Calculate baseline metrics
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    baseline_mse = mean_squared_error(y_test, y_pred_baseline)

    return {
        "Baseline R² Score": baseline_r2,
        "Baseline Mean Absolute Error": baseline_mae,
        "Baseline Mean Squared Error": baseline_mse
    }