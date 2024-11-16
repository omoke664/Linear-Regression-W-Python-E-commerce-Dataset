import pandas as pd


def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data



def get_feature_range(data):
    """Get the min and max values for the features."""
    ranges = {
        'session_length': (data['Avg. Session Length'].min(), data['Avg. Session Length'].max()),
        'time_on_app': (data['Time on App'].min(), data['Time on App'].max()),
        'time_on_website': (data['Time on Website'].min(), data['Time on Website'].max()),
        'length_of_membership':(data['Length of Membership'].min(), data['Length of Membership'].max())
    }

    return ranges