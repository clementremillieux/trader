"""Handle config and logger"""

import os

import logging

import configparser

from dotenv import load_dotenv

load_dotenv("config/.env")

config = configparser.ConfigParser()

config.read("config/app.ini")


class CustomFormatter(logging.Formatter):
    """Custom logger with colored output, aligned messages, and relative paths"""

    RED = "\033[31m"

    GREEN = "\033[92m"

    RESET = "\033[0m"

    ORANGE = "\033[33m"

    def format(self, record):
        """Format log records with color, alignment, and additional context"""
        if record.levelno == logging.INFO:
            level_name = self.GREEN + record.levelname + self.RESET
        elif record.levelno == logging.ERROR:
            level_name = self.RED + record.levelname + self.RESET
        elif record.levelno == logging.WARNING:
            level_name = self.ORANGE + record.levelname + self.RESET
        else:
            level_name = record.levelname

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        relative_path = os.path.relpath(record.pathname, project_root)
        module = record.module
        function_name = record.funcName
        line_no = record.lineno

        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        dynamic_part = (
            f"{level_name} -> [{relative_path}.{module}.{function_name}.{line_no}]"
        )

        # Calculate the padding for alignment
        max_dynamic_length = 140
        padding = max(
            0, max_dynamic_length - len(dynamic_part) - len(timestamp) - 1
        )  # -1 for the space after timestamp

        # Format the message
        message = super(CustomFormatter, self).format(record)

        return f"{timestamp} {dynamic_part}:{' ' * padding} {message}"


def setup_logging():
    """setup the logging logger"""

    logger = logging.getLogger(EnvParam.LOGGER_NAME)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()

    handler.setFormatter(CustomFormatter("%(message)s"))

    logger.setLevel(logging.INFO)

    logger.addHandler(handler)

    logger.propagate = False


def load_param_str_config(section: str, param_name: str) -> str:
    """Load env param from .ini"""

    param: str = config.get(section, param_name)

    return param


def load_param_env_file(name: str) -> str:
    """Load env param from .env"""

    return str(os.getenv(name))


class EnvParam:
    """Handling all the env param in a class"""

    LOGGER_NAME: str = str(
        load_param_str_config(section="app", param_name="LOGGER_NAME")
    )

    VERSION: str = str(load_param_str_config(section="app", param_name="VERSION"))

    APP_NAME: str = str(load_param_str_config(section="app", param_name="APP_NAME"))

    GPT_4_1: str = str(load_param_str_config(section="llm", param_name="GPT_4_1"))

    GPT_4_1_MINI: str = str(
        load_param_str_config(section="llm", param_name="GPT_4_1_MINI")
    )

    EMBEDDING_MODEL: str = str(
        load_param_str_config(section="llm", param_name="EMBEDDING_MODEL")
    )

    CHROMA_PERSIST_DIRECTORY: str = str(
        load_param_str_config(section="chroma", param_name="CHROMA_PERSIST_DIRECTORY")
    )

    CANDIDATE_COLLECION_NAME: str = str(
        load_param_str_config(section="chroma", param_name="CANDIDATE_COLLECION_NAME")
    )

    OPENAI_API_KEY: str = str(load_param_env_file(name="OPENAI_API_KEY"))


env_param = EnvParam()
