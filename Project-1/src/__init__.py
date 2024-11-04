# src/__init__.py

from .data_gathering import fetch_stock_data
from .preprocessing import preprocess_data
from .feature_engineering import add_technical_indicators

__all__ = ["fetch_stock_data", "preprocess_data", "add_technical_indicators"]

PACKAGE_VERSION = "1.0"
DEFAULT_DATA_SOURCE = "yfinance"

# Initialization message
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Initializing package 'src' version {PACKAGE_VERSION}")
