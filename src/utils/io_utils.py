""" 
Production-grade input/output utilities for reading and writing csv, json, joblib and yaml files
"""

# workflow
# ensure directories
# read/write csv files
# read/write yaml files
# read/write joblib files
# read/write json files

# import libraries
import yaml
import json
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ensure directories
def ensure_directory(
        filepath: Union[str, Path]
):
    '''Ensure directories exists, if not, creates a new directory
    
    Args:
        filepath: path to directory
        
    Example:
        ensure_directory('logs')
    '''
    path = Path(filepath)
    try:
        logger.info(f'Creating directory : {path}...')
        Path(path).mkdir(exist_ok=True, parents=True)
        logger.info(f'Directory created or already exists..')

    except Exception as e:
        logger.error(f'Error creating directory : {e}')
        raise

# ==============
# CSV OPERATIONS
# ==============

def read_csv(
        filepath: Union[str, Path],
        **kwargs
) -> pd.DataFrame:
    '''Read data from a csv file
    
    Args:
        filepath: Path to csv file
        **kwargs: Additional parameters
    
    Returns:
        pd.DataFrame
    '''
    path = Path(filepath)
    try:
        if not path.exists():
            logger.error(f'File Not Found! Check filepath and try again')
            raise FileNotFoundError(f'Error: File Not Found!')


        logger.info(f'Reading data from {path}...')
        df = pd.read_csv(path, **kwargs)
        logger.info(f'Data successfully read from CSV file with shape : {df.shape}')
        
        if df.empty:
            logger.error(f'DataFrame is empty!')
            raise pd.errors.EmptyDataError(f'Error: DataFrame is empty')


    except pd.errors.ParserError as e:
        logger.error(f'Error parsing csv file : {e}')
        raise

    except Exception as e:
        logger.error(f'Error : {e}')
        raise

def write_csv(
        df: pd.DataFrame,
        output_path: Union[str, Path],
        **kwargs
):
    '''Write data to a csv file
    
    Args:
        df: data to write to csv file
        output_path: filepath for data to be stored
        **kwargs: Additional parameters
    
    Example:
        write_csv_file(df, output_path, index=False)
    '''
    path = Path(output_path)
    ensure_directory(path)

    try:
        logger.info(f'Writing data to {path}...')
        df.to_csv(path,**kwargs)
        logger.info(f'Data successfully written to {path}..')
    
    except ValueError as e:
        logger.error(f'Error writing data to csv file : {e}')
        raise 

    except Exception as e:
        logger.error(f'Error : {e}')
        raise


# ===============
# JSON OPERATIONS
# ===============

def read_json(
        filepath: Union[str, Path],
        **kwargs
):
    '''Read data from json file
    
    Args:
        filepath: File to json file
        **kwargs: Additional parameters
    '''
    path = Path(filepath)
    if not path.exists():
        logger.info(f'File Not Found! Check filepath and try again')
        raise FileNotFoundError(f'Error: File not found!')
    
    try:
        logger.info(f'Reading data from {path}...')
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file, **kwargs)
            logger.info(f'Data successfully read from {path}')

    except json.JSONDecodeError as e:
        logger.error(f'Error decoding json file : {e}')
        raise 

    except Exception as e:
        logger.error(f'Error : {e}')
        raise


def write_json(
        data: Union[Dict, pd.DataFrame],
        output_path: Union[str, Path],
        **kwargs
):
    '''Write data to a json filee
    
    Args:
        data : Data to write to json file
        output_path : Filepath for data to be stored
        **kwargs : Additional parameters
    '''
    path = Path(output_path)
    ensure_directory(path)

    try:
        logger.info(f'Writing data to {path}...')
        with open(path, 'w') as file:
            json.dump(data, file, **kwargs)
            logger.info(f'Data successfully saved to {path}')
    
    except ValueError as e:
        logger.error(f'Data could not be saved: {e}')
        raise

    except Exception as e:
        logger.error(f'Error : {e}')
        raise


