import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder


class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline for the churn prediction dataset.

    Applies LabelEncoder to all remaining categorical (object-type) columns.
    Tree-based models (CatBoost, RandomForest) work well with label-encoded
    features and avoid the sparsity introduced by one-hot encoding.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config['feature_engineering']

    @staticmethod
    def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply LabelEncoder to every column with dtype == object.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with all object columns label-encoded.
        """
        def _encode(col: pd.Series) -> pd.Series:
            if col.dtype == 'object':
                return pd.Series(LabelEncoder().fit_transform(col), index=col.index)
            return col

        return df.apply(_encode)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the feature engineering pipeline.

        Args:
            df (pd.DataFrame): Preprocessed DataFrame.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df = self.encode_categoricals(df)
        return df
