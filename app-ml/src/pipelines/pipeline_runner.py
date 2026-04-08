import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

import pandas as pd
from typing import Dict, Any
from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline
from pipelines.inference import InferencePipeline
from pipelines.postprocessing import PostprocessingPipeline


class PipelineRunner:
    """
    Orchestrates all stages of the churn prediction ML pipeline.

    Stages:
    - Preprocessing: clean raw data and encode target
    - Feature engineering: label-encode categorical columns
    - Training: Optuna-tuned CatBoostClassifier
    - Inference: load model and predict on test set
    - Postprocessing: evaluate and save results

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        data_manager (DataManager): Handles data I/O and splitting.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager

        self.preprocessing_pipeline = PreprocessingPipeline(config=config)
        self.feature_eng_pipeline = FeatureEngineeringPipeline(config=config)
        self.training_pipeline = TrainingPipeline(config=config)
        self.inference_pipeline = InferencePipeline(config=config)
        self.postprocessing_pipeline = PostprocessingPipeline(config=config)

    def run_training(self) -> None:
        """
        Run the full training pipeline:
        1. Load raw data
        2. Preprocess
        3. Feature engineering
        4. Train/test split (with optional upsampling)
        5. Train model with Optuna tuning
        6. Save model
        """
        raw_path = self.data_manager.get_raw_data_path()
        df = self.data_manager.load_data(raw_path)

        df = self.preprocessing_pipeline.run(df)
        df = self.feature_eng_pipeline.run(df)

        X_train, X_test, y_train, y_test = self.data_manager.split_data(df)

        model = self.training_pipeline.run(X_train, y_train)
        self.postprocessing_pipeline.run_train(model=model)

    def run_inference(self) -> dict:
        """
        Run the full inference pipeline:
        1. Load raw data
        2. Preprocess and feature-engineer
        3. Split into train/test
        4. Predict on test set
        5. Evaluate and return report

        Returns:
            dict: Classification report for the test set.
        """
        raw_path = self.data_manager.get_raw_data_path()
        df = self.data_manager.load_data(raw_path)

        df = self.preprocessing_pipeline.run(df)
        df = self.feature_eng_pipeline.run(df)

        _, X_test, _, y_test = self.data_manager.split_data(df)

        y_pred = self.inference_pipeline.run(X=X_test)
        report = self.postprocessing_pipeline.run_inference(y_pred=y_pred, y_test=y_test)

        return report
