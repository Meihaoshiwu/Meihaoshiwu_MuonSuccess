import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset

# 配置
num_proc = 8
num_proc_load_dataset = num_proc

# 环境变量路径
MUON_BLOCK_MATRIX_EXPERIMENT_DIR = os.getenv("MUON_BLOCK_MATRIX_EXPERIMENT_DIR")
MODEL_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Models")
RESULTS_BASE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Results")
BASE_LOG_PATH = os.path.join(RESULTS_BASE, "Logs/MuonBlockMatrix")
DATA_EXTRACTED_DIR = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Datasets", "extracted_data", "openwebtext")
TOKENIZED_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "TokenizedData")

DATASET_PATH = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Datasets")
DATASET_CACHE = os.path.join(DATASET_PATH, "cache")
MODEL_NAME = "huggyllama/llama-7b"

if __name__ == '__main__':
    print(datasets.__version__)
    
    # 下载并缓存数据集
    dataset = load_dataset("openwebtext", cache_dir=DATASET_CACHE, num_proc=num_proc_load_dataset)
    
    # 划分训练集和验证集
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    
    tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE,
            local_files_only=True
        )
    
    # 分词处理函数
    def process(example):
        # 编码文本，不添加特殊标记
        ids = tokenizer.encode(example['text'], add_special_tokens=False)
        # 手动添加结束标记（LLaMA的EOS）
        ids.append(tokenizer.eos_token_id)
        return {'ids': ids, 'len': len(ids)}
    
    # 分词处理
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    # 写入二进制文件
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(TOKENIZED_CACHE, f'{split}.bin')  # 关键修改
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    print(f"处理完成,文件保存路径: {TOKENIZED_CACHE}")