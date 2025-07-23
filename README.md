# ✈️ Voyage Analytics – ML-Powered Travel Intelligence

A production-ready, full-stack machine learning system designed for the travel industry. It includes:

- 🔁 **Flight Price Regression** using Random Forest  
- 🚻 **Gender Classification** based on first names  
- 🏨 **Hotel Recommendation System** using content-based filtering  

All systems are built with modular ML pipelines, served via APIs or Streamlit apps, and fully integrated into an MLOps pipeline with Jenkins, Docker, and MLflow.

---

## 📂 Project Structure
voyage-analytics-mlops/
│
> ├── regression_model/ # Flight price prediction (ML + API + Jenkins)
> ├── classification_model/ # Name-based gender classifier
> ├── recommendation_model/ # Hotel recommender Streamlit app
> ├── docker-compose.yaml # MLflow + Airflow + Flask container orchestration
> ├── screenshots/ # Demo images used in documentation
> ├── README.md # You're reading it!
> └── .gitignore # Clean git tracking

## 🚀 How to Run

> ✅ Make sure Docker and Python 3.11+ are installed.

🛫 Run Flight Price API
cd regression_model
docker build -t flight-regression-api -f dockerfile.api .
docker run -p 5000:5000 flight-regression-api


🧠 Run Gender Classification API
cd classification_model
docker build -t gender-classifier-api .
docker run -p 5001:5000 gender-classifier-api

🏨 Run Hotel Recommender App
cd recommendation_model
docker build -t hotel-recommender .
docker run -p 8501:8501 hotel-recommender

🧪 Start All Services (MLflow, Airflow, APIs)
docker compose up


🔧 Tools & Stack
Languages: Python 3.11
Libraries: scikit-learn, pandas, joblib, numpy, Flask, Streamlit
MLOps: MLflow, Jenkins, Docker, Airflow
Deployment: Gunicorn, Docker Compose

🏷️ Tags:
mlops
machine-learning
docker
airflow
mlflow
streamlit
flask
regression
recommendation-system
classification
jenkins
travel
