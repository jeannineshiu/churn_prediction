"""
Inference entrypoint:
- Loads configuration
- Runs inference on the test split of the dataset
- Prints the classification report and plots the confusion matrix
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root)

from common.utils import read_config
from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager


if __name__ == "__main__":
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    data_manager = DataManager(config)
    pipeline_runner = PipelineRunner(config=config, data_manager=data_manager)

    print("Starting inference pipeline...")
    report = pipeline_runner.run_inference()

    print("\nInference complete.")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"Macro F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"Accuracy:    {report['accuracy']:.4f}")
