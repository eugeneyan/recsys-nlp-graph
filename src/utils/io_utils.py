import gzip
import pickle
from typing import Any

from src.utils.logger import logger


def save_model(model: Any, model_path: str) -> None:
    """
    Saves model in gzip format

    Args:
        model: Model to be saved
        model_path: Path to save model to

    Returns:
        (None)
    """
    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info('Model saved to {}'.format(model_path))


def load_model(model_path: str) -> Any:
    """
    Loads model from gzip format

    Args:
        model_path: Path to load model from

    Returns:

    """
    with gzip.open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info('Model loaded from: {}'.format(model_path))
    return model
