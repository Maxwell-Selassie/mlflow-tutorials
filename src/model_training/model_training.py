'''Train model using randomforest, xgboost, lightgbm and log them using mlflow'''

import pandas as pd
import numpy as np
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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_logger, write_csv, write_json, write_joblib,Timer,read_csv
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


def train_baseline_models(x_train,x_test,y_train,y_test) -> None:
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

        # cross-validation
        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        model_results = {}
        for model_name,model in models.items():
            logger.info(f'Starting training : {model_name}')
            
            search = GridSearchCV(
                estimator=model['model'],
                param_grid=model['params'],
                cv=cv,
                refit=True,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            search.fit(x_train, y_train)

            # model persistence
            best_model = search.best_estimator_
            write_joblib(best_model, f'models/{model_name}.joblib')

            # make predictions
            preds = best_model.predict(x_train)
            y_pred = best_model.predict(x_test)

            model_results[model_name] = {
                'Best params' : search.best_params_,
                'Best score' : search.best_score_,
                'Train RMSE' : root_mean_squared_error(y_train, preds),
                'Train R2_score' : r2_score(y_train, preds),
                'Test RMSE' : root_mean_squared_error(y_test,y_pred),
                'Test R2_score' : r2_score(y_test,y_pred),
                'Difference' : root_mean_squared_error(y_test,y_pred) - root_mean_squared_error(y_train, preds)
            }


        write_json(model_results,'data/model_results.json')
        return model_results

def rank_models(model_results: Dict[str, Any]) -> Any:
    '''Rank model based on RMSE'''
    if not isinstance(model_results, dict):
        logger.error(f'function expects a dictionary of model results')
        raise TypeError(f'model_results must be a dictionary!')
    
    for name, result in model_results.items():
        if 'Test RMSE' not in result:
            raise KeyError(f"Missing 'Test RMSE' for : {name}")

    best_model = min(
        model_results.items(),
        key=lambda x: x[1]['Test RMSE']
    )

    return {
        'Best Model name' : best_model[0],
        'metrics' : best_model[1]
    }


def main():
    '''Main pipeline for code execution'''

    # load data 
    x_train, x_test, y_train, y_test = load_data()

    # validate data
    validate_data_(x_train,x_test,y_train,y_test)

    # Train baseline models
    model_results = train_baseline_models(x_train, x_test, y_train, y_test)

    # Best model
    print(rank_models(model_results))

if __name__ == '__main__':
    main()
