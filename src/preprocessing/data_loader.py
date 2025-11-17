'''Load dataset from sklearn.dataset'''


import pandas as pd
from sklearn.datasets import load_diabetes
import logging
import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import Timer, write_json, write_csv

logger = logging.getLogger(__name__)

def load_data():
    with Timer('Data Loading', logger):

        try:
            data = load_diabetes

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
                'Memory_usage' : X.memory_usage(deep=True).sum() / (1024 ** 2)
            }
            write_json(dataset_metadata,'data/project_metadata.json')
            write_csv(X, 'data/train_set.csv')
            write_csv(y, 'data/test_set.csv')

            return X, y

        except Exception as e:
            logger.error(f'Error loading dataset : ', exc_info=e)
        