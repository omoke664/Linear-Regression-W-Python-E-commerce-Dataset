import streamlit as st
import numpy as np
import datetime
import pandas as pd
from data import load_data, get_feature_range
from model import load_model, predict
from viz import plot_residuals, plot_feature_coef, plot_model_metrics, plot_eda, plot_heatmap, plot_jointplot_time_on_app, plot_jointplot_time_on_website, plot_pairplot, plot_yearly_spent_distribution
from metrics import calculate_metrics, calculate_residuals, get_baseline_metrics
from streamlit_option_menu import option_menu


#navbar for navigation
with st.sidebar:
    selected = option_menu("Main Menu", ["Model Performance", "Exploratory Data Analysis"],
                           icons = ['bar-chart', 'graph-up'],
                           menu_icon = "cast", default_index = 0)



# Load your dataset and model
data = load_data('Ecommerce_Customers')
feature_ranges = get_feature_range(data)
model = load_model('lr_model.pkl')

# Title of the app 
st.title("Yearly Amount Spent Predictor 🏬")

# Main content area
st.subheader("About this app")
st.write("""This application predicts the yearly amount spent by customers based on several features.
         This application predicts the yearly amount spent by customers based on several features. 
            You can adjust the following parameters to see how they affect the prediction:
            - **Avg. Session Length**: The average length of time a user spends in a session.
            - **Time on App**: The amount of time a user spends on the app.
            - **Time on Website**: The amount of time a user spends on the website.
            - **Length of Membership**: The duration of the user's membership in years.

            Once you have adjusted the parameters, click the "Predict" button to see the predicted yearly amount spent by the customer.
         """)

if selected == "Model Performance":
    st.title("Model Performance Analysis")
# Sidebar for the sliders
    with st.sidebar.expander("Adjust Features", expanded=True):
        # Sliders for each feature using min and max from the dataset
        session_length = st.slider("Avg. Session Length (minutes)",
                                    min_value=feature_ranges['session_length'][0],
                                    max_value=feature_ranges['session_length'][1],
                                    value=(feature_ranges['session_length'][0] + feature_ranges['session_length'][1]) // 2)
        
        time_on_app = st.slider("Time on App (minutes)",
                                min_value=feature_ranges['time_on_app'][0],
                                max_value=feature_ranges['time_on_app'][1],
                                value=(feature_ranges['time_on_app'][0] + feature_ranges['time_on_app'][1]) // 2)
        
        time_on_website = st.slider("Time on Website (minutes)",
                                    min_value=feature_ranges['time_on_website'][0],
                                    max_value=feature_ranges['time_on_website'][1],
                                    value=(feature_ranges['time_on_website'][0] + feature_ranges['time_on_website'][1]) // 2)
        
        length_of_membership = st.slider("Length of Membership (years)",
                                        min_value=feature_ranges['length_of_membership'][0],
                                        max_value=feature_ranges['length_of_membership'][1],
                                        value=(feature_ranges['length_of_membership'][0] + feature_ranges['length_of_membership'][1]) // 2)

    # Prepare the inputs for the model
    input_features = np.array([[session_length, time_on_app, time_on_website, length_of_membership]])



    # Predict button
    if st.button("Predict"):
        prediction = predict(model, input_features)
        # Display the prediction
        st.write(f"Predicted Yearly Amount Spent: $ {prediction[0]:.2f}")
        # Capture the prediction time
        prediction_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_data = {
            "Timestamp": prediction_time,
            "Avg. Session Length (minutes)": session_length,
            "Time on App (minutes)": time_on_app,
            "Time on Website (minutes)": time_on_website,
            "Length of Membership (years)": length_of_membership,
            "Predicted Yearly Amount Spent": prediction[0]
        }

        # Converted to DataFrame for easy export
        report_df = pd.DataFrame([report_data])

        # Display the report in the app
        st.subheader("Prediction Report")
        st.write(report_df)

        # Download report as CSV file
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report", csv, "prediction_report.csv")

        #Calculate residuals and model metrics
        residuals = calculate_residuals(data, model)
        metrics = calculate_metrics(data, model)
        baseline_metrics = get_baseline_metrics(data)

        #Visualizations
        st.subheader("Model Performance Visualizations")
        st.pyplot(plot_residuals(residuals))
        st.pyplot(plot_feature_coef(model,data))
        st.pyplot(plot_model_metrics(metrics, baseline_metrics))

elif selected == "Exploratory Data Analysis":
    #EDA visualizations
    st.title("Exploratory Data Analysis")
    st.subheader("Dataset Overview")
    st.write(data.head())
    st.pyplot(plot_jointplot_time_on_website(data))
    st.pyplot(plot_jointplot_time_on_app(data))
    st.pyplot(plot_heatmap(data))
    st.pyplot(plot_pairplot(data))


        

