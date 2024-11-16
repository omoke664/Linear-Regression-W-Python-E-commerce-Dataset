import pickle
import numpy as np
import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def load_model(model_path):
    """Load the trained model from a pickle file."""
    with open('lr_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, input_features):
    """Make prediction using trained model."""
    return model.predict(input_features)

def get_coefficients(model):
    """Get the coefficients of the trained model."""
    if hasattr(model, 'coef_'):
        return model.coef_
    else:
        raise ValueError("The model does not have coefficients. Ensure it is a linear model.")

def get_feature_names(model, feature_names):
    """Get the feature names for the coefficients."""
    if hasattr(model, 'coef_'):
        return feature_names
    else:
        raise ValueError("The model does not have coefficients. Ensure it is a linear model.")