""" 
Time utilities for tracking time in production
"""

import logging
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

def get_timestamp(format: str = "%Y%m%d%H%M%S") -> str:
    """
    Get current timestamp in format: YYYYMMDD_HHMMSS
    
    Returns:
        Timestamp string
        
    Example:
        >>> get_timestamp()
        '20250108_143025'
    """
    return datetime.now().strftime(format)


def get_date() -> str:
    """
    Get current date in format: YYYYMMDD
    
    Returns:
        Date string
        
    Example:
        >>> get_date()
        '20250108'
    """
    return datetime.now().strftime("%Y%m%d")


def get_datetime_str(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current datetime with custom format.
    
    Args:
        format_str: strftime format string
        
    Returns:
        Formatted datetime string
        
    Example:
        >>> get_datetime_str("%d/%m/%Y")
        '08/01/2025'
    """
    return datetime.now().strftime(format_str)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
        
    Example:
        >>> format_duration(3725.5)
        '1h 2m 5.50s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"

class Timer:
    """
    Context manager for timing code execution.
    
    Example:
        >>> with Timer("Data loading"):
        ...     df = pd.read_csv("data.csv")
        Data loading completed in 2.34s
    """
    def __init__(self, name: str, logging_instance: Optional[logging.Logger] = None):
        '''Initialize Timer instance'''
        self.name = name
        self.logger = logging_instance
        self.start_time = None
        self.end_time = None
        self.time_elapsed = None

    def __enter__(self) -> 'Timer':
            '''start timer'''
            self.start_time = time.time()
            self.logger.info(f'{self.name} started...')
            return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            '''Stop timer and log duration'''
            self.end_time = time.time()
            self.time_elapsed = self.end_time - self.start_time

            if exc_type is None:
                self.logger.info(f'{self.name} completed in {format_duration(self.time_elapsed)}')

            else:
                self.logger.info(f'{self.name} failed after {format_duration(self.time_elapsed)}')


    def get_time_elapsed(self):
        """Get elapsed time in seconds."""
        if self.time_elapsed is None:
            raise RuntimeError("Timer has not completed yet")
        return self.time_elapsed