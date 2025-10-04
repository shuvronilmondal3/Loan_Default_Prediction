from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the entire pipeline (preprocessing + model)
try:
    model = joblib.load("../models/modelFinal.pkl")
except FileNotFoundError:
    print("Model file not found! Please run 'train_and_save_model.py' to create it.")
    exit()

# This route just shows the main form
@app.route('/')
def home():
    return render_template("index.html")

# This route handles the form submission and calculation
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # --- 1. Collect and prepare input data ---
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
            'DEBTINC': float(request.form['DEBTINC']) if request.form['DEBTINC'].strip() else np.nan
        }])

        # --- 2. Make Prediction ---
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        result = "Default" if prediction == 1 else "Not Default"
        confidence = prediction_proba[prediction] * 100
        prediction_text = f"Prediction: {result} (Confidence: {confidence:.2f}%)"

        # --- 3. Redirect to the new result page ---
        # Pass the result to the 'show_result' route
        return redirect(url_for('show_result', prediction_text=prediction_text))

    # Fallback redirect
    return redirect(url_for('home'))

# This new route is responsible for displaying the result page
@app.route('/result')
def show_result():
    # Get the prediction result from the URL query parameter
    prediction_text = request.args.get('prediction_text', 'No result available.')
    return render_template('prediction.html', prediction_text=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)

