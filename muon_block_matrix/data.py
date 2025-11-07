import os
import torch
import torch.multiprocessing as mp

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import Qwen2Tokenizer
from loguru import logger
from typing import List, Optional

from .config import TOKENIZED_CACHE, MODEL_CACHE, OPENWEBTEXT_EXTRACTED, DATASET_CACHE

def load_dataset_by_name(dataset_name: str):
    """加载数据集函数"""
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
        "openwebtext": "Skylion007/openwebtext",
        "wikitext-103": "wikitext",
        "openwebtext-local_txt": "text",
        "openwebtext-100k-local_txt": "arrow",
    }
    
    if dataset_name not in name2path:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "openwebtext-100k-local_txt":
        # 直接读本地 arrow 文件，**不走网络**
        dataset = load_dataset(
            "arrow",
            data_files=f"{DATASET_CACHE}/Elriggs___openwebtext-100k/default/0.0.0/2b7bfd980d5227806de62ac735f40712a4881273/openwebtext-100k-train.arrow",
            split="train",
            cache_dir=DATASET_CACHE,
        )

    elif dataset_name == "openwebtext-local_txt":
        dataset = load_dataset(
            "text",
            data_files=f"{OPENWEBTEXT_EXTRACTED}/*.txt",
            streaming=True,
            cache_dir=DATASET_CACHE,
            trust_remote_code=True,
        )

    else:
        # 其余数据集走缓存/本地检查
        try:
            dataset = load_dataset(
                name2path[dataset_name],
                cache_dir=DATASET_CACHE,
                local_files_only=True
            )
        except Exception as e:
            raise FileNotFoundError(
                f"数据集 {dataset_name} 在缓存目录 {DATASET_CACHE} 中未找到。"
                f"请确保数据集已下载到缓存目录。错误详情: {e}"
            )
    
    return dataset

class ExperimentPreparer:
    def __init__(
        self,
        texts: List[str],
        tokenizer_name: str,
        cache_dir: str,
        output_file: str,
        batch_size: int = 500,
    ):
        self.texts = texts
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.output_file = output_file
        self.num_workers = min(mp.cpu_count(), 4)
        self.batch_size = batch_size

    @staticmethod
    def _tokenize_worker(worker_data):
        worker_id, text_batches, tokenizer_name, cache_dir = worker_data
        try:
            # 首先尝试从缓存加载
            tokenizer = Qwen2Tokenizer.from_pretrained(
                tokenizer_name, 
                cache_dir=cache_dir,
                local_files_only=True  # 只从本地加载
            )
        except OSError:
            # 如果缓存中没有，则下载（在跳板机上运行）
            print(f"⚠️  缓存中未找到 {tokenizer_name}，开始下载...")
            tokenizer = Qwen2Tokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=cache_dir  # 下载到缓存目录
            )
            print(f"✅ 已下载到缓存: {cache_dir}")

        all_tokens = []
        for text_batch in text_batches:
            encoded = tokenizer.batch_encode_plus(
                text_batch,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            for seq in encoded:
                all_tokens.extend(seq)
        return worker_id, all_tokens, len(text_batches)

    @staticmethod
    def _distribute_data(texts, num_workers, batch_size):
        total = len(texts)
        per = total // num_workers
        rem = total % num_workers
        worker_data = []
        for wid in range(num_workers):
            start = wid * per + min(wid, rem)
            end = start + per + (1 if wid < rem else 0)
            batches = [texts[i : i + batch_size] for i in range(start, end, batch_size)]
            worker_data.append((wid, batches))
        return worker_data

    @staticmethod
    def _merge_and_save(partial_dir, num_workers, output_file):
        all_tokens = []
        for wid in range(num_workers):
            f = os.path.join(partial_dir, f"worker_{wid}.pt")
            if os.path.exists(f):
                data = torch.load(f, weights_only=True)
                all_tokens.extend(data["tokens"])
        torch.save(all_tokens, output_file)
        return all_tokens

    def tokenize(self):
        if os.path.exists(self.output_file):
            logger.info(f"Cache hit -> {self.output_file}")
            return torch.load(self.output_file, weights_only=True)

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # 根据 worker 数量选择处理方式
        if self.num_workers > 1:
            return self._tokenize_multi_process()
        else:
            return self._tokenize_single_process()

    def _tokenize_multi_process(self):
        """多进程处理"""
        partial_dir = f"{self.output_file}_partial"
        os.makedirs(partial_dir, exist_ok=True)
        logger.info("Using multi-process tokenization start")
        # 1. 数据划分
        worker_data = self._distribute_data(self.texts, self.num_workers, self.batch_size)
        worker_inputs = [(wid, batches, self.tokenizer_name, self.cache_dir)
                        for wid, batches in worker_data]

        # 2. 并行 tokenize
        results = []
        with mp.Pool(self.num_workers) as pool:
            for wid, tokens, _ in pool.imap_unordered(
                self._tokenize_worker, worker_inputs
            ):
                torch.save({"tokens": tokens}, os.path.join(partial_dir, f"worker_{wid}.pt"))
                results.append((wid, len(tokens)))

        # 3. 合并 & 清理
        all_tokens = self._merge_and_save(partial_dir, self.num_workers, self.output_file)
        for wid, _ in results:
            try:
                os.remove(os.path.join(partial_dir, f"worker_{wid}.pt"))
            except FileNotFoundError:
                pass
        os.rmdir(partial_dir)
        logger.info(f"Multi-process tokenization finished -> {len(all_tokens)} tokens")
        return all_tokens

    def _tokenize_single_process(self):
        """单进程处理"""
        logger.info("Using single-process tokenization")
        
        # 修正：将 texts 分成批次，与多进程模式保持一致
        batches = [self.texts[i:i + self.batch_size] for i in range(0, len(self.texts), self.batch_size)]
        worker_input = (0, batches, self.tokenizer_name, self.cache_dir)
        
        # 直接调用 worker 函数
        _, all_tokens, _ = self._tokenize_worker(worker_input)
        
        # 保存结果
        torch.save(all_tokens, self.output_file)
        logger.info(f"Single-process tokenization finished -> {len(all_tokens)} tokens")
        return all_tokens


class MoonDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        max_length: int = 512,
    ):
        self.dataset_name = dataset_name
        self.max_length = max_length
        
        # 直接加载预处理好的 tokenized 数据
        self.cache_file = os.path.join(TOKENIZED_CACHE, f"{dataset_name}.bin")
        if not os.path.exists(self.cache_file):
            raise FileNotFoundError(
                f"预处理文件不存在: {self.cache_file}\n"
                f"请先运行: python -m muon_block_matrix.preprocess --dataset {dataset_name}"
            )
        
        self.tokens = torch.load(self.cache_file, weights_only=True)
        print(f"📁 加载预处理数据: {len(self.tokens)} tokens")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        return torch.tensor(self.tokens[start:end], dtype=torch.long)