"""logger_config.py"""

import logging
from config.config import setup_logging, EnvParam

setup_logging()

logger = logging.getLogger(EnvParam.LOGGER_NAME)
