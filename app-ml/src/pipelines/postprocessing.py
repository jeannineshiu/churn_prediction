import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import classification_report
from common.utils import save_model, plot_results


class PostprocessingPipeline:
    """
    Postprocessing pipeline for churn prediction.

    Responsibilities:
    - Saving the trained model after training
    - Evaluating and displaying inference results
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run_train(self, model: Any) -> None:
        """
        Save the trained model to the path specified in config.

        Args:
            model: Trained CatBoostClassifier (or compatible sklearn estimator).
        """
        model_path = self.config['pipeline_runner']['model_path']
        save_model(model, base_path=model_path)

    def run_inference(self, y_pred: np.ndarray, y_test: pd.Series) -> dict:
        """
        Evaluate predictions and display classification report + confusion matrix.

        Args:
            y_pred (np.ndarray): Predicted labels.
            y_test (pd.Series): True labels.

        Returns:
            dict: Classification report as a dictionary.
        """
        plot_results(y_test, y_pred, save_path="inference_results.png")
        report = classification_report(y_test, y_pred, output_dict=True)
        return report
