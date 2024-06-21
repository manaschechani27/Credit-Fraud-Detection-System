import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load(r'C:\Users\91952\Desktop\Credit card fraud detection\Data\random_forest_model.pkl')

# Example of the comma-separated string of test input
test_input_str = "-1.1666691913432188,1.32197869570702,-0.575195465659831,0.395503410710055,-0.551812648075679,-1.17234488601789,-1.03615163993916,-0.417695816571423,-0.228769752797668,-0.819610105594849,0.664582858885129,-0.508754125710815,-0.234832219022152,0.160393724037626,0.126382810115988,0.994191659189235,-1.16302302287649,-0.0649714223609736,0.886025666644818,-0.768040506226172,-0.410389489652245,-0.468704866684149,-0.977936022323078,0.113434525485654,0.396269278978682,0.0775036838145536,0.901030068656141,-0.0630111192738049,0.0171331292367211,-0.17775338991488668"

# Convert the string to a list of floats
test_input = [float(val) for val in test_input_str.split(',')]

# Define the feature names used during training (ensure the order matches)
feature_names = ['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                 'Amount']

# Create a DataFrame from the test input
test_df = pd.DataFrame([test_input], columns=feature_names)

# Make predictions
predictions = model.predict(test_df)

# Print predictions
print("Predicted Class:", predictions[0])
