"""
Training entrypoint:
- Loads configuration
- Runs the full training pipeline (preprocessing, feature engineering, training, postprocessing)
- Saves the trained model to the models folder
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

    print("Starting training pipeline...")
    pipeline_runner.run_training()
    print("Training complete. Model saved.")
