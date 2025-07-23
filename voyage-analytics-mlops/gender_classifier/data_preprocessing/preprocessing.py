# preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ================================
# 1. Load the Users Dataset
# ================================
print("Loading dataset...")
df = pd.read_csv("data/users.csv")

print("\nInitial data snapshot:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nGender value counts (including 'none'):")
print(df['gender'].value_counts(dropna=False))

# =========================================
# 2. Clean the Dataset - Remove 'none'
# =========================================
print("\nCleaning dataset...")

# Filter only rows where gender is 'male' or 'female'
df = df[df['gender'].isin(['male', 'female'])].copy()
df['gender'] = df['gender'].str.lower()

print("\nCleaned gender distribution:")
print(df['gender'].value_counts())

# =========================================
# 3. Feature Engineering
# =========================================

# Feature 1: Length of the name
df['name_length'] = df['name'].apply(len)

# Feature 2: Does the name include a middle name?
df['has_middle_name'] = df['name'].apply(lambda x: 1 if len(x.strip().split()) > 2 else 0)

# Feature 3: Encode the company column
le_company = LabelEncoder()
df['company_encoded'] = le_company.fit_transform(df['company'])

# =========================================
# 4. Prepare Features and Labels
# =========================================

# Selected features for training
features = ['age', 'name_length', 'has_middle_name', 'company_encoded']
X = df[features]

# Target label
y = df['gender']

# Encode target variable
le_gender = LabelEncoder()
y_encoded = le_gender.fit_transform(y)  # male → 1, female → 0

# =========================================
# 5. Train/Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nTrain/Test split completed.")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# =========================================
# 6. Save Encoders and Data (Optional)
# =========================================

# Save encoders for use in inference/API
import joblib

joblib.dump(le_company, "data_preprocessing/company_encoder.pkl")
joblib.dump(le_gender, "data_preprocessing/gender_encoder.pkl")

# Save preprocessed datasets
X_train.to_csv("data_preprocessing/X_train.csv", index=False)
X_test.to_csv("data_preprocessing/X_test.csv", index=False)
pd.Series(y_train).to_csv("data_preprocessing/y_train.csv", index=False)
pd.Series(y_test).to_csv("data_preprocessing/y_test.csv", index=False)

print("\nPreprocessing completed successfully!")
