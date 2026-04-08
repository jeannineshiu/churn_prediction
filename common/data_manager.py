import os
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class DataManager:
    """
    Handles all data I/O and splitting operations for the churn prediction pipeline.

    Responsibilities:
    - Loading raw CSV data
    - Splitting data into train/test sets with optional upsampling
    - Saving predictions
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataManager with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load data from a CSV or Parquet file.

        Args:
            path (str): Path to the file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        if path.endswith('.csv'):
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def get_raw_data_path(self) -> str:
        """Return the full path to the raw data file."""
        return os.path.join(
            self.config['data_manager']['data_folder'],
            self.config['data_manager']['raw_data_name']
        )

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets with optional minority-class upsampling.

        Args:
            df (pd.DataFrame): Full preprocessed and feature-engineered DataFrame.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        target_col = self.config['preprocessing']['target_column']
        test_size = self.config['pipeline_runner']['test_size']
        random_state = self.config['pipeline_runner']['random_state']

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        if self.config['pipeline_runner'].get('upsample', False):
            X_train, y_train = self._upsample(X_train, y_train, random_state)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def _upsample(
        X_train: pd.DataFrame, y_train: pd.Series, random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Upsample the minority class to match the majority class size.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            random_state (int): Random seed.

        Returns:
            Tuple of upsampled (X_train, y_train).
        """
        train_df = pd.concat([X_train, y_train], axis=1)
        target_col = y_train.name

        majority = train_df[train_df[target_col] == 0]
        minority = train_df[train_df[target_col] == 1]

        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=random_state
        )

        balanced = pd.concat([majority, minority_upsampled])
        return balanced.drop(columns=[target_col]), balanced[target_col]

    @staticmethod
    def save_predictions(y_pred: pd.Series, y_test: pd.Series, path: str) -> None:
        """
        Save predictions alongside true labels to a CSV file.

        Args:
            y_pred (pd.Series): Predicted labels.
            y_test (pd.Series): True labels.
            path (str): Output file path.
        """
        results = pd.DataFrame({
            'true_label': y_test.values,
            'predicted_label': y_pred
        })
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results.to_csv(path, index=False)
        print(f"Predictions saved to {path}")
