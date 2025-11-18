'''Production-grade ML training pipeline with MLflow tracking and model registry'''

import pandas as pd
import numpy as np
import time
import logging
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
from sklearn.model_selection import GridSearchCV, KFold
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
from utils import setup_logger, write_csv, write_json, write_joblib, Timer, format_duration
from preprocessing import load_data

logger = setup_logger('model_training', 'logs/')

# Constants
EXPERIMENT_NAME = 'Diabetes Prediction Model(Regression)'
MLFLOW_TRACKING_URI = 'file:./mlruns'


def validate_data_(x_train, x_test, y_train, y_test) -> None:
    '''Validate input data integrity'''
    try:
        logger.info('Starting data validation...')
        
        # Length checks
        if len(x_train) != len(y_train) or len(x_test) != len(y_test):
            raise ValueError('Data length mismatch!')
        
        # Type checks
        if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
            raise ValueError('Targets must be pd.Series!')
        
        # Null checks
        if x_train.isnull().any().any() or x_test.isnull().any().any():
            raise ValueError('Training data contains nulls!')
        
        logger.info(' Data validation passed')
        
    except Exception as e:
        logger.error(f'Data validation failed: {e}', exc_info=True)
        sys.exit(1)


def train_models(x_train, x_test, y_train, y_test) -> Dict[str, Dict[str, Any]]:
    '''Train and evaluate multiple models with MLflow tracking'''
    
    with Timer('model_training', logger):
        # Configure MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        client = MlflowClient()
        
        models = {
            'randomforest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [80, 100, 120],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 8]
                }
            },
            'lightgbm': {
                'model': LGBMRegressor(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [80, 100],
                    'max_depth': [5, 7],
                    'reg_lambda': [0.3, 0.5],
                    'learning_rate': [0.05, 0.1]
                }
            },
            'xgboost': {
                'model': XGBRegressor(objective='reg:squarederror', random_state=42),
                'params': {
                    'n_estimators': [80, 100, 120],
                    'max_depth': [2, 5, 7],
                    'reg_lambda': [0.05, 0.1, 0.5],
                    'learning_rate': [0.1, 0.3, 0.5]
                }
            }
        }
        
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        model_results = {}
        
        for model_name, model_config in models.items():
            with mlflow.start_run(run_name=f'model_{model_name}'):
                try:
                    # Set tags
                    mlflow.set_tag('Developer', 'Maxwell Selassie')
                    mlflow.set_tag('dataset', 'sklearn.datasets.diabetes')
                    mlflow.set_tag('model_type', model_name)
                    mlflow.set_tag('purpose', 'hyperparameter_tuning')
                    
                    logger.info(f'Training {model_name}...')
                    
                    # Grid search
                    search = GridSearchCV(
                        estimator=model_config['model'],
                        param_grid=model_config['params'],
                        cv=cv,
                        refit=True,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    start_time = time.time()
                    search.fit(x_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Log best params
                    for param, value in search.best_params_.items():
                        mlflow.log_param(param, value)
                    
                    # Save model
                    best_model = search.best_estimator_
                    write_joblib(best_model, f'models/{model_name}.joblib')
                    
                    # Predictions
                    train_preds = best_model.predict(x_train)
                    test_preds = best_model.predict(x_test)
                    
                    # Metrics
                    train_rmse = root_mean_squared_error(y_train, train_preds)
                    test_rmse = root_mean_squared_error(y_test, test_preds)
                    train_r2 = r2_score(y_train, train_preds)
                    test_r2 = r2_score(y_test, test_preds)
                    
                    model_results[model_name] = {
                        'Training_Time': format_duration(training_time),
                        'Best_Params': search.best_params_,
                        'CV_RMSE': abs(search.best_score_),
                        'Train_RMSE': train_rmse,
                        'Test_RMSE': test_rmse,
                        'Train_R2': train_r2,
                        'Test_R2': test_r2,
                        'Overfit_Gap': test_rmse - train_rmse
                    }
                    
                    # Log metrics to MLflow
                    mlflow.log_metric('cv_rmse', model_results[model_name]['CV_RMSE'])
                    mlflow.log_metric('train_rmse', train_rmse)
                    mlflow.log_metric('test_rmse', test_rmse)
                    mlflow.log_metric('train_r2', train_r2)
                    mlflow.log_metric('test_r2', test_r2)
                    mlflow.log_metric('overfit_gap', test_rmse - train_rmse)
                    mlflow.log_metric('training_time_seconds', training_time)
                    
                    # Feature importance
                    perm_imp = permutation_importance(
                        best_model, x_test, y_test, 
                        n_repeats=10, random_state=42
                    )
                    
                    feature_importance = pd.DataFrame({
                        'Feature': x_train.columns,
                        'Importance': perm_imp.importances_mean,
                        'Std': perm_imp.importances_std
                    }).sort_values('Importance', ascending=False)
                    
                    feat_imp_path = f'data/feature_importance/{model_name}_importance.csv'
                    write_csv(feature_importance, feat_imp_path, index=False)
                    mlflow.log_artifact(feat_imp_path)
                    
                    # Log CV results
                    cv_results = pd.DataFrame(search.cv_results_)
                    cv_path = f'data/cv_results/{model_name}_cv_results.csv'
                    write_csv(cv_results, cv_path, index=False)
                    mlflow.log_artifact(cv_path)
                    
                    # Log model with signature
                    signature = infer_signature(x_train, train_preds)
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        name='model',
                        signature=signature,
                        input_example=x_train.head(5),
                        registered_model_name=f'diabetes_{model_name}'
                    )
                    
                    logger.info(f' {model_name} completed: Test RMSE = {test_rmse:.4f}')
                    
                except Exception as e:
                    logger.error(f'Training failed for {model_name}: {e}', exc_info=True)
                    continue
        
        # Save comparison
        comparison_df = pd.DataFrame(model_results).T.sort_values('Test_RMSE')
        write_csv(comparison_df, 'data/model_comparison.csv')
        write_json(model_results, 'data/model_results.json', indent=4)
        
        logger.info(f'\n{comparison_df.to_string()}')
        
        return model_results


def leaderboard(experiment_name: str = EXPERIMENT_NAME, 
                metric: str = 'test_rmse') -> pd.DataFrame:
    '''Display and return ranked model leaderboard'''
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        logger.error(f'Experiment "{experiment_name}" not found')
        return pd.DataFrame()
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f'metrics.{metric} ASC']
    )
    
    if len(runs) == 0:
        logger.warning('No runs found')
        return pd.DataFrame()
    
    print(f'\n{"="*80}')
    print(f'LEADERBOARD: {experiment_name}')
    print(f'{"="*80}')
    print(f'{"Model":<15} | {"Test RMSE":<12} | {"Test R²":<12} | {"Run ID":<36}')
    print(f'{"-"*80}')
    
    leaderboard_data = []
    for run in runs:
        model_type = run.data.tags.get('model_type', 'unknown')
        test_rmse = run.data.metrics.get('test_rmse', float('inf'))
        test_r2 = run.data.metrics.get('test_r2', 0.0)
        
        print(f'{model_type:<15} | {test_rmse:<12.4f} | {test_r2:<12.4f} | {run.info.run_id}')
        
        leaderboard_data.append({
            'Model': model_type,
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2,
            'Run_ID': run.info.run_id
        })
    
    print(f'{"="*80}\n')
    
    return pd.DataFrame(leaderboard_data)

def promote_best_model(experiment_name: str = EXPERIMENT_NAME, 
                    model_name: str = 'Diabetes_Prediction_Model', 
                    metrics: str = 'Test_RMSE'):
    '''Promote best performing model to production'''
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.error(f'Experiment "{experiment_name}" not found!')
        return pd.DataFrame()
    
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f'metric.{metrics} ASC'],
        max_results=1
    )[0]

    if best_run is None:
        logger.error(f'Experiment "{experiment_name}" has no runs')

    best_run_id = best_run.info.run_id
    model_uri = f'runs:/{best_run_id}/model'

    # register a new version
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage='Production'
    )

    print(f'Version {mv.version} of {model_name} registered to Production')


def main():
    '''Main execution pipeline'''
    
    # Load data
    x_train, x_test, y_train, y_test = load_data()
    
    # Validate
    validate_data_(x_train, x_test, y_train, y_test)
    
    # Train models
    results = train_models(x_train, x_test, y_train, y_test)
    
    # Show leaderboard
    leaderboard_df = leaderboard()
    
    # Identify best model
    if not leaderboard_df.empty:
        best = leaderboard_df.iloc[0]
        logger.info(f'Best Model: {best["Model"]} (RMSE: {best["Test_RMSE"]:.4f})')

    promote_best_model()


if __name__ == '__main__':
    main()