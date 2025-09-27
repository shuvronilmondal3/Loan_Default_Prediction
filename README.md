# Loan Default Prediction Web Application


Overview
This project is an end-to-end machine learning application designed to predict the likelihood of a customer defaulting on a home equity loan. It addresses a critical business need for financial institutions by automating credit risk assessment, making the process faster, more consistent, and less prone to human bias.

The application is built using a Random Forest Classifier trained on the HMEQ (Home Equity Mortgage Qualification) dataset. It is deployed as a user-friendly web interface using the Flask framework, allowing for real-time predictions based on customer inputs.

Features
Real-Time Predictions: Get an instant "Default" or "Not Default" prediction.

User-Friendly Interface: A simple and intuitive web form for entering customer data.

Robust Preprocessing: The backend pipeline automatically handles missing values and encodes categorical data before making a prediction.

Scalable Architecture: The separation of the training script and the Flask app makes the model easy to retrain and update.

Tech Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, Pandas, NumPy

Model Serialization: Joblib

Frontend: HTML, Basic CSS

Project Structure
For the application to run correctly, your project folder must be organized as follows:

/loan-default-prediction
|
|-- app.py                  # Main Flask application
|-- train_and_save_model.py # Script to train and save the ML model
|-- hmeq.csv                # The dataset file
|-- requirements.txt        # List of Python dependencies
|-- README.md               # You are here!
|
|-- /models/
|   |-- modelRF.pkl         # The serialized model pipeline (created by the training script)
|
|-- /templates/
    |-- index.html          # The HTML template for the web form

Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
Python 3.7 or higher

pip (Python package installer)

2. Clone the Repository
git clone [https://github.com/your-username/loan-default-prediction.git](https://github.com/your-username/loan-default-prediction.git)
cd loan-default-prediction

3. Create a Virtual Environment (Recommended)
It's best practice to create a virtual environment to manage project dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

4. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

5. Download the Dataset
Download the hmeq.csv dataset and place it in the root directory of the project. You can find the dataset on Kaggle.

6. Train the Model
Before you can run the web app, you need to train the model and create the modelRF.pkl file. Run the training script from the root directory:

python train_and_save_model.py

This will create the /models folder and save the trained pipeline inside it.

7. Run the Flask Application
Now you are ready to start the web server.

python app.py

The application will now be running. Open your web browser and navigate to:
https://www.google.com/search?q=http://127.0.0.1:5000

How to Use
Open the web application in your browser.

Fill in all the fields in the form with the customer's loan application details.

Click the "Predict" button.

The application will display the prediction result ("Default" or "Not Default") along with the model's confidence level.

The Modeling Process
The machine learning model was built using the following workflow:

Data Loading: The hmeq.csv dataset was loaded using pandas.

Preprocessing:

Imputation: Missing numerical values were filled with the mean of their respective columns, and missing categorical values were filled with the most frequent value.

One-Hot Encoding: Categorical features (REASON, JOB) were converted into a numerical format that the model can understand.

Pipeline Creation: All preprocessing steps were bundled together with the RandomForestClassifier into a single Scikit-learn Pipeline. This ensures that the exact same transformations are applied to both the training data and any new data submitted via the web app.

Model Training & Serialization: The pipeline was trained on the data and then saved to a single file (modelRF.pkl) using joblib for easy loading in the Flask application.
