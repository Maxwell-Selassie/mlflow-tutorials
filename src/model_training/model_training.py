'''Train model using randomforest, xgboost, lightgbm and log them using mlflow'''

import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Dict,Optional,Union
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sys

sys.path.insert(0, Path(__file__).parent.parent)
from utils import setup_logger, write_csv, write_json, write_joblib,Timer,read_csv
from preprocessing import load_data

logger = setup_logger('model_training', 'logs/')


def _load_data_():
    '''Load feature and target datasets to be used for training'''
    try:
        logger.info(f'Collecting feature and target sets from source...')
        x_train, x_test, y_train, y_test = load_data()
        logger.info(f'Datasets successfully loaded!')
        return x_train, x_test, y_train, y_test
    
    except ValueError as e:
        logger.error(f'Failed to successfully collect and load datasets')
        sys.exit(1)

    except Exception as e:
        logger.error(f'Error collecting and loading datasets : {e}')
        sys.exit(1)

def _validate_data_(x_train, x_test, y_train, y_test):
    '''Validate the shape and length of individual sets to avoid downstream errors'''
    try:
        logger.info(f'Starting data validation..')
        if len(x_train) != len(y_train) and len(x_test) != len(y_test):
            logger.warning(f'Data validation failed! Check the length of individual datasets')
            raise ValueError(f'Data validation failed!')
        
        if not isinstance(y_train, pd.Series) and isinstance(y_test, pd.DataFrame):
            logger.warning(f'Data Validation failed! (y_train and y_test) must have single columns')
            raise ValueError(f'Data validation failed!')
        
    except Exception as e:
        logger.error(f'Could not perform data validation!')
        sys.exit(1)


