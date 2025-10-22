from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

DATA_DIR = 'data'
TRAINED_MODEL_DIR = 'models'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, 'model.pkl')
TARGET_COL = 'company_size'

# Load training data
df = pd.read_csv(TRAIN_PATH)

# Separate features and target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Train Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X, y)

if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)

joblib.dump(logreg, MODEL_PATH)
print(f"Logistic Regression model trained and saved to {MODEL_PATH}")