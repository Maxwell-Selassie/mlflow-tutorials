'''Load dataset from sklearn.dataset'''


import pandas as pd
from sklearn.datasets import load_diabetes
import logging
import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import Timer, write_json, write_csv, setup_logger, ensure_directory


logger = setup_logger('preprocessing', 'logs/')


def load_data():
    with Timer('Data Loading', logger):

        try:
            data = load_diabetes()

            X = pd.DataFrame(data.data,
                            columns=data.feature_names)
            
            y = pd.Series(data.target, name='target')

            dataset_metadata = {
                'Project_name' : 'Diabetes Prediction Model',
                'Author' : 'Maxwell Selassie Hiamatsu',
                'Version' : 1,
                'Number of observations' : len(X),
                'Number of features' : len(X.columns),
                'Feature list' : X.columns.tolist(),
                'Memory_usage' : f'{((X.memory_usage(deep=True).sum() + y.memory_usage(deep=True)) / (1024 ** 2)):.2f}MB'
            }
            write_json(dataset_metadata,'data/project_metadata.json')
            logger.info(f'Project Metadata saved!')

            write_csv(X, 'data/train_set.csv', index=False)
            logger.info(f'Training set data saved!')

            write_csv(y, 'data/target_set.csv',index=False)
            logger.info(f'Target set data saved!')

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return x_train, x_test, y_train, y_test

        except Exception as e:
            logger.error(f'Error loading dataset : ', exc_info=e)
            sys.exit(1)

        