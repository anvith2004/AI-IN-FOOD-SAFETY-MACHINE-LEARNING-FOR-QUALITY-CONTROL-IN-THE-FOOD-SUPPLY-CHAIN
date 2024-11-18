# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import json

# Import custom_preprocessor from preprocessing.py
from preprocessing import custom_preprocessor

# Load the trained pipeline and label encoder
with open('xgboost_model.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the dropdown options
with open('dropdown_options.json', 'r') as f:
    dropdown_options = json.load(f)

# Initialize Flask app
app = Flask(__name__)

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html', dropdown_options=dropdown_options)

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = {
            "category": request.form['category'],
            "type": request.form['type'],
            "subject": request.form['subject'],
            "notifying_country": request.form['notifying_country'],
            "classification": request.form['classification'],
            "operator": request.form['operator'],
            "origin": request.form['origin'],
            "hazards": request.form['hazards']
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Predict
        prediction = model_pipeline.predict(input_df)
        decoded_prediction = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "success": True,
            "predicted_risk_decision": decoded_prediction
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
