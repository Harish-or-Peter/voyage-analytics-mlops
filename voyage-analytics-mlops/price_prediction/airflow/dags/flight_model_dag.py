from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import os


def train_model():
    df = pd.read_csv('/opt/airflow/data/flights.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df.drop(columns=['travelCode', 'userCode', 'date'], inplace=True)

    X = df.drop('price', axis=1)
    y = df['price']

    cat_cols = ['from', 'to', 'flightType', 'agency']
    num_cols = ['time', 'distance', 'month', 'day', 'weekday']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough')

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.set_tracking_uri("http://mlflow:5000")  # Or mlruns folder
    mlflow.set_experiment("flight_price_airflow")

    with mlflow.start_run():
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, '/opt/airflow/artifacts/flight_price_model.pkl')

with DAG("flight_model_training", start_date=datetime(2023,1,1), schedule_interval="@daily", catchup=False) as dag:
    task1 = PythonOperator(
        task_id="train_and_log_model",
        python_callable=train_model
    )
