import pickle
import streamlit as st
import numpy as np
import pandas as pd


#loading the trained model
with open('lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

#Title of the app
st.title("E-Commerce Linear Model Predictor")

# user input
df = pd.read_csv('Ecommerce_Customers')


#