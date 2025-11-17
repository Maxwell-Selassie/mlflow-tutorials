""" 
Production-grade logging utility with dedicated file and console logging system
"""

# import libraries
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from logging.handlers import RotatingFileHandler
from typing import Union, Optional
import sys

def setup_logger(
        name: str,
        log_dir: Union[str, Path] ,
        backupcount: int = 7,
        max_bytes: int = 10485760,
        log_level: str ='INFO',
        log_format: Optional[str] = None,
        date_format: Optional[str] = None,
        console_log: bool = True 
) -> logging.Logger:
    """
    Setup production-grade logger with file rotation and console output.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        log_format: Custom log format string
        date_format: Custom date format string
        console_output: Whether to output to console
        file_permissions: Unix file permissions for log files
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("eda_pipeline", "../logs/", log_level="INFO")
        >>> logger.info("Pipeline started")
    """
    # create logging instance
    logger = logging.Logger(name)

    if logger.handlers:
        return logger

    # set log level 
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level_obj)

    logger.propagate = False

    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    if date_format is None:
        date_format = '%Y-%m-%d %H:%M:%S'

    formatter = logging.Formatter(log_format, date_format)

    log_file = log_dir/f'{name}.log'
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backupcount
    )

    file_handler.setLevel(log_level_obj)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler(
            sys.stdout
        )
        console_handler.setLevel(log_level_obj)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
