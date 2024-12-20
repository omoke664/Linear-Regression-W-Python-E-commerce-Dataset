import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from model import load_model, load_data





def plot_residuals(residuals):
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins = 30, kde = True)
    plt.title("Residuals Distribution")
    plt.xlabel = ("Residuals")
    plt.xlabel = ("Frequency")
    plt.grid()
    return plt

def plot_yearly_spent_distribution(data):
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Yearly Amount Spent'], bins=30, kde=True)
    plt.title("Distribution of Yearly Amount Spent")
    plt.xlabel("Yearly Amount Spent")
    plt.ylabel("Frequency")
    plt.grid()
    return plt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_model_metrics(metrics, baseline_metrics):
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values()),
        'Baseline': list(baseline_metrics.values())
    })

    plt.figure(figsize= (10, 5))
    sns.barplot(x = 'Metric', y = 'Value', data= metrics_df, color = 'blue', label= 'Model Metrics')
    sns.barplot(x = 'Metric', y = 'Baseline', data=metrics_df, color = 'orange', label = 'Baseline Metrics', alpha = 0.5)
    plt.title('Model Metrics vs Baseline Metrics')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    return plt


def plot_eda(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot = True, fmt = ".2f", cmap = 'coolwarm', square= True)
    plt.title('Correlation Heatmap')
    plt.grid()
    return plt

def plot_jointplot_time_on_website(data):
    plt.figure(figsize=(10,8))
    sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = data, alpha = 0.5)
    plt.title("Joint Plot: Time on Website vs Yearly Amount Spent")
    return plt

def plot_jointplot_time_on_app(data):
    plt.figure(figsize=(10, 8))
    sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=data, alpha=0.5)
    plt.title("Joint Plot: Time on App vs Yearly Amount Spent")
    return plt

def plot_heatmap(data):
    plt.figure(figsize=(10, 5))
    sns.heatmap(data.select_dtypes(float).corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Heatmap")
    return plt

def plot_pairplot(data):
    plt.figure(figsize=(10, 10))
    sns.pairplot(data, kind='scatter', plot_kws={"alpha": 0.4})
    plt.title("Pair Plot of Features")
    return plt

def plot_feature_coef(model, data):
    X = data[['Avg. Session Length', 'Time on Website', 'Time on App', 'Length of Membership']]
    cdf = pd.DataFrame(model.coef_, X.columns, columns=['Coef'])
    cdf.sort_values(by='Coef').plot(kind='bar', figsize=(10, 6))
    plt.title('Feature Coefficients')
    return plt
