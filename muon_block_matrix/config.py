import os

import traceback
from datetime import datetime

# === 自定义存储路径 ===
MUON_BLOCK_MATRIX_EXPERIMENT_DIR = os.getenv("MUON_BLOCK_MATRIX_EXPERIMENT_DIR")
# === 基于环境变量的存储路径 ===
MODEL_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Models")
DATASET_PATH = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Datasets") 
TOKENIZED_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "TokenizedData")
RESULTS_BASE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Results")
OPENWEBTEXT_EXTRACTED = os.path.join(DATASET_PATH, "openwebtext_extracted")
DATASET_CACHE = os.path.join(DATASET_PATH, "cache")
DATASET_DOWNLOAD = os.path.join(DATASET_PATH, "download")

def log_diagnostic(module_name, message=""):
    """简单的诊断日志函数"""
    pid = os.getpid()
    ppid = os.getppid()
    
    # 创建诊断日志目录
    log_dir = f"{RESULTS_BASE}/muon_diagnostics"
    os.makedirs(log_dir, exist_ok=True)
    
    # 每个进程有自己的日志文件
    log_file = f"{log_dir}/process_{pid}.log"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] PID:{pid} PPID:{ppid} | {module_name} | {message}\n")
        
        # 记录调用栈
        stack = traceback.extract_stack()
        for _, frame in enumerate(stack[:-2]):  # 排除诊断函数本身
            f.write(f"    {frame.filename}:{frame.lineno} in {frame.name}\n")
        f.write("\n")

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
        matrix_block_num=4,      # 要正交化的矩阵分块数量
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
        self.matrix_block_num = matrix_block_num
    
    def __repr__(self):
        return (f"ExperimentConfig(step_func_name={self.step_func_name}, "
                f"model_name={self.model_name}, dataset_name={self.dataset_name}, "
                f"hidden_size={self.hidden_size}, max_tokens={self.max_tokens}, "
                f"max_epochs={self.max_epochs}, lr={self.lr}, wd={self.wd})")

class StreamConfig:
    """流式数据配置"""
    def __init__(
        self,
        batches_per_chunk: int = 1024,
        num_buffers: int = 16,
        num_loaders: int = 4,
        max_length: int = 512,
        batch_size: int = 32,
        loader_rest_threshold: float = 0.8,
        world_size: int =8, # 缓冲区被多少个训练进程读取
    ):
        assert (batch_size % world_size) == 0
        self.batches_per_chunk = batches_per_chunk
        self.num_buffers = num_buffers
        self.num_loaders = num_loaders
        self.max_length = max_length
        self.batch_size = batch_size
        self.loader_rest_threshold = loader_rest_threshold
        self.world_size = world_size