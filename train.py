import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_DIR = 'data'
TRAINED_MODEL_DIR = 'models'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
RF_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, 'model.pkl')
TARGET_COL = 'company_size'

# Load training data
df = pd.read_csv(TRAIN_PATH)

# Separate features and target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]


# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)

joblib.dump(rf, RF_MODEL_PATH)
print(f"Random Forest model trained and saved to {RF_MODEL_PATH}")