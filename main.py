from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model, scaler, and column names
rf_classifier = joblib.load('rf_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')
column_names = joblib.load('column_names.pkl')

@app.route('/')
def home():
    return "Welcome to the ML Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame(data)

    # Ensure correct columns and preprocessing
    input_data = pd.get_dummies(input_data, drop_first=True)
    for col in set(column_names) - set(input_data.columns):
        input_data[col] = 0
    input_data = input_data[column_names]

    # Scale the data
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = rf_classifier.predict(input_data)
    return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    app.run(debug=True)