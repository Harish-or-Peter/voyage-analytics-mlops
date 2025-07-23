#app.py

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("gender_classifier.pkl")
company_encoder = joblib.load("data_preprocessing/company_encoder.pkl")
gender_encoder = joblib.load("data_preprocessing/gender_encoder.pkl")

app = Flask(__name__)

# Home page with HTML form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            name = request.form["name"]
            company = request.form["company"]
            age = int(request.form["age"])

            # Preprocessing
            name_length = len(name)
            has_middle_name = 1 if len(name.strip().split()) > 2 else 0
            try:
                company_encoded = company_encoder.transform([company])[0]
            except ValueError:
                company_encoded = -1

            input_df = pd.DataFrame([{
                "age": age,
                "name_length": name_length,
                "has_middle_name": has_middle_name,
                "company_encoded": company_encoded
            }])

            # Prediction
            pred = model.predict(input_df)[0]
            gender_label = gender_encoder.inverse_transform([pred])[0]

            return render_template("index.html", prediction=gender_label, name=name, age=age, company=company)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
