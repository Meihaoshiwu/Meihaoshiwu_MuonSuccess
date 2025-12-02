import os

# === 自定义存储路径 ===
MUON_BLOCK_MATRIX_EXPERIMENT_DIR = os.getenv("MUON_BLOCK_MATRIX_EXPERIMENT_DIR")
# === 基于环境变量的存储路径 ===
MODEL_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Models")
DATASET_PATH = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Datasets") 
TOKENIZED_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "TokenizedData")
RESULTS_BASE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Results")
DATA_EXTRACTED_DIR = os.path.join(DATASET_PATH, "extracted_data")
DATASET_CACHE = os.path.join(DATASET_PATH, "cache")
DATASET_DOWNLOAD = os.path.join(DATASET_PATH, "download")

# 原始常量
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_WD = 0.1
DEFAULT_LOSS_THRESHOLD = 3.0
DEFAULT_MAX_EPOCHS = 10

class ExperimentConfig():
    def __init__(
        self,
        step_func_name,
        step_func:callable,
        log_file_path,
        model_name="qwen",
        dataset_name="openwebtext",
        optimizer_name="muon",
        batch_size=DEFAULT_BATCH_SIZE,
        hidden_size=DEFAULT_BATCH_SIZE,
        max_position_embeddings=2048,
        max_length=DEFAULT_MAX_LENGTH,
        loss_threshold=DEFAULT_LOSS_THRESHOLD,
        max_epochs=DEFAULT_MAX_EPOCHS,
        lr=DEFAULT_LR,
        wd=DEFAULT_WD, #AdamW使用
        sampler=None,
        max_tokens=None,  # 新增：最大token数量
        ):
        self.step_func_name = step_func_name
        self.step_func=step_func
        self.log_file_path = log_file_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.max_length = max_length
        self.loss_threshold = loss_threshold
        self.max_epochs = max_epochs
        self.lr = lr
        self.wd = wd
        self.sampler = sampler
        self.max_tokens = max_tokens or float('inf')  # 默认无限制
    
    def __repr__(self):
        return (f"ExperimentConfig(step_func_name={self.step_func_name}, "
                f"model_name={self.model_name}, dataset_name={self.dataset_name}, "
                f"hidden_size={self.hidden_size}, max_tokens={self.max_tokens}, "
                f"max_epochs={self.max_epochs}, lr={self.lr}, wd={self.wd})")