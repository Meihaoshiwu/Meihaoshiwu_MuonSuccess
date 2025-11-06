__version__ = "0.1.0"
__name__ = "sdt"

from .config import ExperimentConfig
from .data import ExperimentPreparer, MoonDataset
from .model import create_qwen_model
from .train import run_experiment
from .optimizer import Muon, get_optimizer, STEP_MAP

__all__ = [
    "ExperimentConfig",
    "ExperimentPreparer", 
    "MoonDataset",
    "create_qwen_model",
    "run_experiment",
    "Muon",
    "get_optimizer",
    "STEP_MAP",
]