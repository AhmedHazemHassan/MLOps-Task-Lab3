import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow

mlflow.sklearn.autolog()
mlflow.set_experiment("Salaries Company Size Classification")

# Load preprocessed train and test data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

TARGET_COL = 'company_size'
X_train = train_df.drop(TARGET_COL, axis=1)
y_train = train_df[TARGET_COL]
X_test = test_df.drop(TARGET_COL, axis=1)
y_test = test_df[TARGET_COL]

n_estimators_list = [50, 100, 200]
max_depth_list = [5, 10, 20]

print("Starting hyperparameter tuning with MLflow...")

with mlflow.start_run(run_name="RF Tuning Parent"):
    for n_Estimators in n_estimators_list:
        for Max_depth in max_depth_list:
            with mlflow.start_run(run_name=f"RF n={n_Estimators} d={Max_depth}", nested=True):
                model = RandomForestClassifier(n_estimators=n_Estimators, max_depth=Max_depth, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mlflow.log_param("n_estimators", n_Estimators)
                mlflow.log_param("max_depth", Max_depth)
                mlflow.log_metric("precision_macro", precision_score(y_test, y_pred, average='macro'))
                mlflow.log_metric("recall_macro", recall_score(y_test, y_pred, average='macro'))
                mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average='macro'))
                print(f"Trained RF with n_estimators={n_Estimators}, max_depth={Max_depth}")

print("Hyperparameter tuning completed.")