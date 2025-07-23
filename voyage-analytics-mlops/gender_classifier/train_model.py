# train_model.py

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

# ============================
# 1. Load Preprocessed Data
# ============================
X_train = pd.read_csv("data_preprocessing/X_train.csv")
X_test = pd.read_csv("data_preprocessing/X_test.csv")
y_train = pd.read_csv("data_preprocessing/y_train.csv").squeeze()
y_test = pd.read_csv("data_preprocessing/y_test.csv").squeeze()

# ============================
# 2. Set MLflow Experiment
# ============================
mlflow.set_experiment("gender-classifier-exp")

with mlflow.start_run(run_name="RandomForest_v1"):

    # ============================
    # 3. Train Classifier
    # ============================
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # ============================
    # 4. Evaluate the Model
    # ============================
    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['female', 'male']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nF1 Score: {f1:.4f}")

    # Optional: Cross-validation score
    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"\nCross-validated F1 score: {cv_score.mean():.4f}")

    # ============================
    # 5. MLflow Logging
    # ============================
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("cv_f1_score", cv_score.mean())

    # Log model artifact
    joblib.dump(clf, "gender_classifier.pkl")
    mlflow.sklearn.log_model(clf, "model", registered_model_name="GenderClassifierModel")

    print("\nModel saved as gender_classifier.pkl")
