from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the model (assuming you saved your trained model as 'model.pkl')
model = joblib.load('model.pkl')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

# Define a route for the home page
@app.route('/')
def home():
    return 'Welcome to the Online Food Order Prediction API'

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
