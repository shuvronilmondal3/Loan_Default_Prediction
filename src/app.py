from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load pipeline (preprocessing + model)
model = joblib.load("../models/modelFinal.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect inputs
        input_data = pd.DataFrame([{
            'LOAN': float(request.form['LOAN']),
            'MORTDUE': float(request.form['MORTDUE']),
            'VALUE': float(request.form['VALUE']),
            'REASON': request.form['REASON'],
            'JOB': request.form['JOB'],
            'YOJ': float(request.form['YOJ']),
            'DEROG': float(request.form['DEROG']),
            'DELINQ': float(request.form['DELINQ']),
            'CLAGE': float(request.form['CLAGE']),
            'NINQ': float(request.form['NINQ']),
            'CLNO': float(request.form['CLNO']),
            'DEBTINC': float(request.form['DEBTINC']) if request.form['DEBTINC'].strip() != "" else np.nan
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        result = "Default" if prediction == 1 else "No Default"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
