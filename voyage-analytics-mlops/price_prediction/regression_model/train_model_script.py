import pandas as pd

# Load data
flights = pd.read_csv('flights.csv')


flights['date'] = pd.to_datetime(flights['date'])
flights['month'] = flights['date'].dt.month
flights['day'] = flights['date'].dt.day
flights['weekday'] = flights['date'].dt.weekday

# Drop non-useful columns
flights.drop(columns=['travelCode', 'userCode', 'date'], inplace=True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_cols = ['from', 'to', 'flightType', 'agency']
numerical_cols = ['time', 'distance', 'month', 'day', 'weekday']

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = flights.drop('price', axis=1)
y = flights['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

import joblib
joblib.dump(pipeline, 'flight_price_model.pkl')

