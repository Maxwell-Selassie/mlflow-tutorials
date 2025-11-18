'''Train model using randomforest, xgboost, lightgbm and log them using mlflow'''

import pandas as pd
import numpy as np
import time
import logging
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Dict,Optional,Union,Any
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_logger, write_csv, write_json, write_joblib,Timer,read_csv, format_duration
from preprocessing import load_data

logger = setup_logger('model_training', 'logs/')


def validate_data_(x_train, x_test, y_train, y_test) -> None:
    '''Validate the shape and length of individual sets to avoid downstream errors'''
    try:
        logger.info(f'Starting data validation..')
        if len(x_train) != len(y_train) or len(x_test) != len(y_test):
            logger.warning(f'Data validation failed! Check the length of individual datasets')
            raise ValueError(f'Data validation failed!')
        
        if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
            logger.warning(f'Data Validation failed! (y_train and y_test) must have single columns')
            raise ValueError(f'Data validation failed!')
        
    except Exception as e:
        logger.error(f'Could not perform data validation!')
        sys.exit(1)


def train_models(x_train,x_test,y_train,y_test) -> None:
    '''Train the baseline models'''

    with Timer('model_training',logger):
        models = {
            'randomforest': {
            'model': RandomForestRegressor(random_state=42),
            'params' : {
                'n_estimators' : [80, 100, 120],
                'max_depth' : [3, 5, 7],
                'min_samples_split' : [2, 5, 8]
            }
            },
            'lightgbm' : {
            'model' : LGBMRegressor(random_state=42),
            'params' : {
                'n_estimators' : [80, 100, 120],
                'max_depth' : [2, 5, 7],
                'num_leaves' : [31, 40, 51],
                'reg_lambda' : [0.1, 0.3, 0.5],
                'learning_rate' : [0.05, 0.1, 0.5]
            }
            },
            'xgboost' : {
            'model' : XGBRegressor(objective='reg:squarederror', random_state=42),
            'params' : {
                'n_estimators' : [80, 100, 120],
                'max_depth' : [2, 5, 7],
                'reg_lambda' : [0.05, 0.1, 0.5],
                'learning_rate' : [0.1, 0.3, 0.5]
            }
            }
        }

        mlflow.set_experiment(f'Diabetes Prediction Model(Regression)')
        client = MlflowClient()

        # cross-validation
        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        model_results = {}
        for model_name,model in models.items():
            with mlflow.start_run(run_name=f'model_{model_name}'):

                mlflow.set_tag('Developer', 'Maxwell Selassie')
                mlflow.set_tag('dataset', 'sklearn.dataset')
                mlflow.set_tag('model_type', model_name)
                mlflow.set_tag('purpose', 'hyper-parameter tuning')

                logger.info(f'Starting training : {model_name}')

                mlflow.log_param('params', model['params'])
                
                search = GridSearchCV(
                    estimator=model['model'],
                    param_grid=model['params'],
                    cv=cv,
                    refit=True,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )

                try:
                    start_time = time.time()
                    search.fit(x_train, y_train)
                    time_elapsed = time.time() - start_time
                except Exception as e:
                    logger.error(f'Training failed for model : {model_name} : {e}')
                    continue

                # log best params
                for p, v in search.best_params_.items():
                    mlflow.log_param(p,v)

                # model persistence
                best_model = search.best_estimator_
                write_joblib(best_model, f'models/{model_name}.joblib')

                # make predictions
                preds = best_model.predict(x_train)
                y_pred = best_model.predict(x_test)

                

                model_results[model_name] = {
                    'Training Time' : format_duration(time_elapsed),
                    'Best_params' : search.best_params_,
                    'Best_score' : abs(search.best_score_),
                    'Train_RMSE' : root_mean_squared_error(y_train, preds),
                    'Train_R2_score' : r2_score(y_train, preds),
                    'Test_RMSE' : root_mean_squared_error(y_test,y_pred),
                    'Test_R2_score' : r2_score(y_test,y_pred),
                    'Difference' : root_mean_squared_error(y_test,y_pred) - root_mean_squared_error(y_train, preds)
                }
                
                mlflow.log_metric('Best_score', model_results[model_name]['Best_score'])
                mlflow.log_metric('Train_rmse', model_results[model_name]['Train_RMSE'])
                mlflow.log_metric('Test_rmse', model_results[model_name]['Test_RMSE'])
                mlflow.log_metric('Train_R2_score', model_results[model_name]['Train_R2_score'])
                mlflow.log_metric('Test_R2_score', model_results[model_name]['Test_R2_score'])

                signature = infer_signature(
                    x_train, 
                    best_model.predict(x_train),
                    params=search.best_params_
                )

                mlflow.sklearn.log_model(
                    sk_model=search.best_estimator_,
                    name='model',
                    signature=signature,
                    input_example=x_train.head(5)
                )

                # permutation importance
                perm_importance = permutation_importance(
                    estimator=best_model,
                    X=x_train,
                    y=y_train,
                    n_repeats=10,
                    random_state=42
                )

                feature_importance = pd.DataFrame(
                    {
                        'Features' : x_train.columns,
                        'Importances' : perm_importance.importances_mean
                    }
                ).sort_values(by='Importances', ascending=False)
                feature_imp_path = f'artifacts/{model_name}_feature_importance.csv'
                write_csv(feature_importance, feature_imp_path)

                mlflow.log_artifact(feature_imp_path)


            write_json(model_results,'data/model_results.json', indent=4)

def leaderboard(experiment_name: str = 'Diabetes Prediction Model(Regression)', metric: int = 'Test_rmse') -> None:
    '''Model model performances based on Best Score'''
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f'metric.{metric} ASC']
    )

    print(f'\nLEADERBOARD (lowest is better)\n')
    for r in runs:
        print(
            f'Run ID: {r.info.run_id:<20} | Model: {r.data.tags.get('model_type')}' 
            f'Best Score: {r.data.metrics.get('Test_rmse',0.0):.4f} | Best R2_score: {r.data.metrics.get('Test_R2_score',0.0):.4f}'
        )
    


def main():
    '''Main pipeline for code execution'''

    # load data 
    x_train, x_test, y_train, y_test = load_data()

    # validate data
    validate_data_(x_train,x_test,y_train,y_test)

    # Train baseline models
    train_models(x_train, x_test, y_train, y_test)

    # Model ranking
    leaderboard()


if __name__ == '__main__':
    main()
