import numpy as np
import optuna
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier


class TrainingPipeline:
    """
    Training pipeline for churn prediction using CatBoostClassifier and Optuna.

    Uses StratifiedKFold cross-validation to tune hyperparameters, then trains
    a final model on the full training set with the best configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary with training parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config['training']
        self.optuna_config = self.config.get('optuna', {})
        self.search_space = self.config['optuna']['search_space']

    def tune_hyperparams(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[CatBoostClassifier, optuna.Study]:
        """
        Run Optuna hyperparameter search using StratifiedKFold cross-validation,
        then train a final model on the full training data with the best parameters.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            Tuple of (trained CatBoostClassifier, completed Optuna Study).
        """
        np.random.seed(self.config['random_seed'])
        ss = self.search_space
        n_splits = self.config['n_cv_splits']
        cfg = self.config

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", ss["learning_rate"]["low"], ss["learning_rate"]["high"]
                ),
                "depth": trial.suggest_int(
                    "depth", ss["depth"]["low"], ss["depth"]["high"]
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", ss["l2_leaf_reg"]["low"], ss["l2_leaf_reg"]["high"]
                ),
                "iterations": cfg["iterations"],
                "loss_function": cfg["loss_function"],
                "eval_metric": cfg["eval_metric"],
                "verbose": 0,
            }

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg['random_seed'])
            f1_scores = []
            best_iterations = []

            for train_idx, val_idx in cv.split(X_train, y_train):
                x_tr, x_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = CatBoostClassifier(**params, random_seed=cfg['random_seed'], allow_writing_files=False)
                model.fit(
                    x_tr, y_tr,
                    eval_set=(x_val, y_val),
                    early_stopping_rounds=cfg['early_stopping_rounds'],
                    use_best_model=True,
                    verbose=False
                )

                y_pred = model.predict(x_val)
                f1_scores.append(f1_score(y_val, y_pred, average='macro'))
                best_iterations.append(model.get_best_iteration())

            avg_best_iter = int(np.mean(best_iterations))
            trial.set_user_attr("best_iteration", avg_best_iter)
            return float(np.mean(f1_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config['random_seed'])
        )
        study.optimize(objective, n_trials=self.optuna_config["n_trials"])

        best_params = study.best_trial.params.copy()
        best_params.update({
            "iterations": study.best_trial.user_attrs["best_iteration"],
            "loss_function": self.config["loss_function"],
            "verbose": 0,
        })

        final_model = CatBoostClassifier(
            **best_params,
            random_seed=self.config['random_seed'],
            allow_writing_files=False
        )
        final_model.fit(X_train, y_train, verbose=False)

        return final_model, study

    def run(self, X_train: pd.DataFrame, y_train: pd.Series) -> CatBoostClassifier:
        """
        Run the full training pipeline.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            CatBoostClassifier: Trained model.
        """
        model, study = self.tune_hyperparams(X_train, y_train)
        print(f"Best F1 (macro, CV): {study.best_trial.value:.4f}")
        print(f"Best params: {study.best_trial.params}")
        return model
