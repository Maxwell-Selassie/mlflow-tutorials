import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

data = load_diabetes()
x = data.data
y = data.target 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

mlflow.set_experiment('diabetes_regression')
with mlflow.start_run(run_name='rf_baseline'):

    n_estimators = 200
    max_depth = 5
    random_state = 42

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('random_state', random_state)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(x_train, y_train)

    preds = model.predict(x_test)

    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('r2', r2)

    Path('artifacts').mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.scatter(y_test, preds)
    plt.xlabel('True labels')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. Actual')
    plot_path = 'artifacts/pred_vs_true.png'
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path)

    mlflow.sklearn.log_model(model, 'model')