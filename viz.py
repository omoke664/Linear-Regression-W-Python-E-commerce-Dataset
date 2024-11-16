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

def plot_coef(model, data):
    """Plot the coefficients of the model."""
    if hasattr(model, 'coef_'):
        # Check if the model has a single output
        if model.coef_.ndim == 1:
            coefficients = model.coef_
        else:
            # If the model has multiple outputs, take the first set of coefficients
            coefficients = model.coef_[0]

        # Ensure the data is loaded correctly
        # Assuming 'data' is already passed as a DataFrame, no need to load it again
        # data = load_data('Ecommerce_Customers.csv')  # Uncomment if needed

        features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
        
        # Create a DataFrame for plotting
        cdf = pd.DataFrame(coefficients, index=features, columns=['Coef'])

        # Sort the DataFrame by coefficients for better visualization
        cdf = cdf.sort_values(by='Coef', ascending=False)

        # Plotting
        plt.figure(figsize=(10, 5))
        sns.barplot(x=cdf.index, y='Coef', data=cdf, palette='viridis')
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        plt.title('Feature Coefficients')
        plt.grid(axis='y')
        plt.tight_layout()  
        return plt
    else:
        raise ValueError("Model does not have coefficients.")


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
    plt.figure(figsize=(10, 5))
    sns.heatmap(data.corr(), annot = True, fmt = ".2f", cmap = 'coolwarm', square= True)
    plt.title('Correlation Heatmap')
    plt.grid()
    return plt

def plot_jointplot_time_on_website(data):
    plt.figure(figsize=(10,5))
    sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = data, alpha = 0.5)
    plt.title("Joint Plot: Time on Website vs Yearly Amount Spent")
    return plt

def plot_jointplot_time_on_app(data):
    plt.figure(figsize=(10, 5))
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