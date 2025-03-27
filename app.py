from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("student_model.pkl")

# Feature names matching the dataset
FEATURES = ["Hours Studied", "Previous Scores", "Extracurricular Activities", 
            "Sleep Hours", "Sample Question Papers Practiced"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form  # Get form data

        # Convert form data into a DataFrame
        input_data = pd.DataFrame([[float(data[feature]) for feature in FEATURES]], columns=FEATURES)

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({'prediction': round(prediction, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
