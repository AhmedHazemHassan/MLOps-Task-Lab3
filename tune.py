
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow

mlflow.sklearn.autolog()  # Enable autologging for scikit-learn
mlflow.set_experiment("Salaries Company Size Classification")

# Load preprocessed train and test data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

TARGET_COL = 'company_size'
X_train = train_df.drop(TARGET_COL, axis=1)
y_train = train_df[TARGET_COL]
X_test = test_df.drop(TARGET_COL, axis=1)
y_test = test_df[TARGET_COL]

with mlflow.start_run(run_name="RandomForest Autolog"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # For multiclass, use average='macro' for fair comparison
    mlflow.log_metric("precision_macro", precision_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("recall_macro", recall_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average='macro'))
