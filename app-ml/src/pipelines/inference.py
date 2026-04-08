import numpy as np
import pandas as pd
from typing import Dict, Any
from common.utils import load_model


class InferencePipeline:
    """
    Inference pipeline for churn prediction.

    Loads a trained CatBoostClassifier and generates predictions for the given input.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self, X: pd.DataFrame) -> np.ndarray:
        """
        Load the trained model and predict churn labels for X.

        Args:
            X (pd.DataFrame): Feature DataFrame for inference.

        Returns:
            np.ndarray: Predicted class labels (0 = No Churn, 1 = Churn).
        """
        model = load_model(base_path=self.config['pipeline_runner']['model_path'])
        y_pred = model.predict(X)
        return y_pred
