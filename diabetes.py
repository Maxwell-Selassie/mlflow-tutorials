import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def leaderboard(experiment_name='diabetes_advanced_regression', metric='rmse'):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f'metrics.{metric}ASC'],
    )
    print('\nLEADERBOARD (lowest RMSE first) \n')
    for r in runs:
        print(
            f'Run: {r.info.run_id} | Model: {r.data.tags.get('model_type')}'
            f'| RMSE: {r.data.metrics.get('rmse'):.4f} | R2: {r.data.metrics.get('r2'):.4f}')

data = load_diabetes()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

params_grid = {
    'randomforest': {
        'model' : RandomForestClassifier(random_state=42),
        'params' : {
            'n_estimators' :[90,120],
            'max_depth' : [4,6]
        }
    },
    'xgboost' : {
        'model' : XGBRegressor(objective='reg:squarederror', random_state=42),
        'params' : {
            'n_estimators' : [90, 120],
            'max_depth' : [4,7],
            'learning_rate' : [0.1, 0.3]
        }
    },
    'lightgbm' : {
        'model' : LGBMRegressor(random_state=42),
        'params' : {
            'n_estimators' : [90,120],
            'max_depth' : [0, 2],
            'learning_rate' : [0.08,0.1],
            'num_leaves' : [31, 35]
        }
    }
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

mlflow.set_experiment('diabetes_advanced_regression')
client = MlflowClient()

for model_name, config in params_grid.items():
    with mlflow.start_run(run_name=f'{model_name}_gridsearchCV'):

        print(f'Running Model : {model_name}...')
        
        base_model = config['model']
        params = config['params']

        mlflow.set_tag('developer','Maxwell Selassie')
        mlflow.set_tag('dataset','sklearn.diabetes')
        mlflow.set_tag('model_type',model_name)
        mlflow.set_tag('purpose','hyperparameter tuning')

        mlflow.log_param('params_grid', params)

        # gridsearchCV
        model = GridSearchCV(
            estimator=base_model,
            param_grid=params,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            refit=True
        )

        model.fit(x_train, y_train)

        best_model = model.best_estimator_
        best_params = model.best_params_
        
        print(f'best params : {best_params}')

        # log best params
        for p, v in best_params.items():
            mlflow.log_param(p,v)




        preds = best_model.predict(x_test)

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

        # model signature and input example
        signature = infer_signature(
            x_train, best_model.predict(x_train)
        )
        input_example = x_train.head(5)

        # log model with signature
        mlflow.sklearn.log_model(
            sk_model=best_model, artifact_path='model',
            signature=signature, input_example=input_example
        )

        print(f'Logged {model_name} to MLflow')

leaderboard()