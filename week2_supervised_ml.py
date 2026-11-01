"""
Yuva Intern - Week 2: Supervised Machine Learning Models
Regression: Diabetes dataset (Linear Regression, Random Forest Regressor)
Classification: Breast Cancer Wisconsin dataset (Logistic Regression, Decision Tree)
Run: python week2_supervised_ml.py
Dependencies: numpy, pandas, scikit-learn
"""
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

def run_regression():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.20, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=42
        )
    }
    print("\n=== REGRESSION RESULTS (Diabetes dataset) ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        print(f"\n{name}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

def run_classification():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.20,
        random_state=42, stratify=data.target
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42)
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=42, max_depth=4
        )
    }
    print("\n=== CLASSIFICATION RESULTS (Breast Cancer dataset) ===")
    print("Class 0 = malignant; Class 1 = benign")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"\n{name}")
        print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
        print(f"Precision: {precision_score(y_test, predictions):.4f}")
        print(f"Recall:    {recall_score(y_test, predictions):.4f}")
        print(f"F1-score:  {f1_score(y_test, predictions):.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    run_regression()
    run_classification()
