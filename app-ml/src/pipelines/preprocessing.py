import numpy as np
import pandas as pd
from typing import Dict, Any


class PreprocessingPipeline:
    """
    Preprocessing pipeline for the churn prediction dataset.

    Steps:
    - Fix TotalCharges (empty strings → NaN → float)
    - Drop rows with tenure == 0 (customers who never used the service)
    - Drop irrelevant columns (e.g. customerID)
    - Encode target variable (Yes/No → 1/0)

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config['preprocessing']

    @staticmethod
    def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace empty/whitespace TotalCharges entries with NaN and convert to float.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with TotalCharges as float.
        """
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        return df

    @staticmethod
    def drop_zero_tenure(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where tenure == 0 (customers who never used the service).

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame without zero-tenure rows.
        """
        df = df.drop(labels=df[df['tenure'] == 0].index, axis=0)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): Columns to drop.

        Returns:
            pd.DataFrame: DataFrame with columns removed.
        """
        existing = [c for c in columns if c in df.columns]
        return df.drop(columns=existing)

    @staticmethod
    def encode_target(df: pd.DataFrame, target_col: str, mapping: dict) -> pd.DataFrame:
        """
        Map target column values to numeric labels.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Name of the target column.
            mapping (dict): Mapping from string labels to integers (e.g. {'Yes': 1, 'No': 0}).

        Returns:
            pd.DataFrame: DataFrame with encoded target column.
        """
        df[target_col] = df[target_col].map(mapping)
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.

        Args:
            df (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = self.fix_total_charges(df)
        df = self.drop_zero_tenure(df)
        df = self.drop_columns(df, self.config['drop_columns'])
        df = self.encode_target(df, self.config['target_column'], self.config['target_mapping'])
        return df
