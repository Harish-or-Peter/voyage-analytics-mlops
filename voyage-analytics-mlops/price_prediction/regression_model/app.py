from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load model
model = joblib.load("flight_price_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "Flight Price Prediction API is up!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame from input
        df = pd.DataFrame([data])

        # Predict using model
        prediction = model.predict(df)[0]

        return jsonify({"predicted_price": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
