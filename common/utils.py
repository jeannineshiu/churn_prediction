import yaml
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, Any
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix


def read_config(path: Union[str, Path]) -> dict:
    """
    Reads a YAML configuration file and returns it as a dictionary.

    Args:
        path (str or Path): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = __name__, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with optional file output and standard formatting.

    Args:
        name (str): Logger name.
        log_file (Optional[str]): If provided, logs will also be written to this file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def save_model(model: Any, base_path: str) -> None:
    """
    Save model as .cbm if CatBoost, .pkl if sklearn.

    Args:
        model: Trained model.
        base_path (str): File path without extension.
    """
    path = Path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, CatBoostClassifier):
        model.save_model(str(path.with_suffix(".cbm")))
        print(f"Saved CatBoost model to {path.with_suffix('.cbm')}")
    elif isinstance(model, BaseEstimator):
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(model, f)
        print(f"Saved sklearn model to {path.with_suffix('.pkl')}")
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def load_model(base_path: str) -> Any:
    """
    Load model by checking both .cbm and .pkl variants.

    Args:
        base_path (str): File path without extension.

    Returns:
        Loaded model.
    """
    path = Path(base_path)
    cbm_path = path.with_suffix(".cbm")
    pkl_path = path.with_suffix(".pkl")

    if cbm_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(cbm_path))
        return model
    elif pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Neither {cbm_path} nor {pkl_path} found.")


def plot_results(y_test: pd.Series, y_pred: pd.Series, save_path: str = "inference_results.png") -> None:
    """
    Plot confusion matrix and print classification report.

    Args:
        y_test (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        save_path (str): Path to save the confusion matrix figure.
    """
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")
