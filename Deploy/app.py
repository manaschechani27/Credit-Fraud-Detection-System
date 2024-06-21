from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load(r'C:\Users\91952\Desktop\Credit card fraud detection\Data\random_forest_model.pkl')

# Define feature names used during training
feature_names = ['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                 'Amount']

# Define a function to preprocess input data
def preprocess_input(input_data):
    # Convert comma-separated string to list of floats
    input_list = [float(val) for val in input_data.split(',')]
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_list], columns=feature_names)
    return input_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = request.form['input_data']
    
    # Preprocess the input data
    input_df = preprocess_input(input_data)
    
    # Make predictions
    prediction = model.predict(input_df)[0]
    
    # Return prediction as JSON response
    return jsonify({'prediction': int(prediction)})

@app.route('/redirect')
def redirect_page():
    return render_template('report.html')

if __name__ == '__main__':
    app.run(debug=True)
