import os
import torch
import torch.multiprocessing as mp

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import Qwen2Tokenizer
from loguru import logger
from typing import List, Optional

from .config import TOKENIZED_CACHE, DATA_EXTRACTED_DIR, DATASET_CACHE

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
            data_files=f"{DATASET_CACHE}/openwebtext-100k-train.arrow",
            split="train",
            cache_dir=DATASET_CACHE,
        )

    elif dataset_name == "openwebtext-local_txt":
        output_dir = os.path.join(DATA_EXTRACTED_DIR, "openwebtext")
        dataset = load_dataset(
            "text",
            data_files=f"{output_dir}/*.txt",
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
        model_cache_dir: str,
        output_file: str,
        batch_size: int = 500,
        shard_dirs: Optional[List[str]] = None,
    ):
        self.texts = texts
        self.tokenizer_name = tokenizer_name
        self.model_cache_dir = model_cache_dir
        self.output_file = output_file
        self.num_workers = min(mp.cpu_count(), 16)
        self.batch_size = batch_size
        self.shard_dirs = shard_dirs
        
        if shard_dirs:
            print(f"🔧 使用分片目录模式: {len(shard_dirs)} 个分片")
            self.mode = "sharded"
        else:
            print(f"🔧 使用文本列表模式: {len(texts)} 个文本")
            self.mode = "text_list"
    
    # ==================== 分片目录模式相关函数 ====================
    
    @staticmethod
    def _tokenize_shard_worker(worker_data):
        """处理一个分片目录的worker函数，自己保存结果"""
        worker_id, shard_dir, tokenizer_name, cache_dir, batch_size, save_path = worker_data
        import time
        
        pid = os.getpid()
        print(f"[进程{worker_id}|PID:{pid}] 🔄 开始处理分片: {shard_dir}")
        
        # 加载tokenizer
        start_time = time.time()
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(
                tokenizer_name, 
                cache_dir=cache_dir,
                local_files_only=True
            )
        except (TypeError, OSError):
            print(f"[进程{worker_id}|PID:{pid}] ⚠️  缓存中未找到 {tokenizer_name}，开始下载...")
            tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        
        print(f"[进程{worker_id}|PID:{pid}] ✅ tokenizer加载完成，耗时: {time.time()-start_time:.2f}秒")
        
        # 使用load_dataset加载分片目录下的所有文件
        print(f"[进程{worker_id}|PID:{pid}] 📂 使用load_dataset加载分片文件...")
        dataset_start = time.time()
        
        try:
            # 使用load_dataset加载文本文件
            dataset = load_dataset(
                "text",
                data_files=f"{shard_dir}/*.txt",
                streaming=False,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            
            # 正确处理load_dataset返回的数据结构
            if isinstance(dataset, dict):  # 如果是DatasetDict
                # 获取第一个split（通常是"train"）
                split_name = list(dataset.keys())[0]
                texts = dataset[split_name]["text"]
            else:
                # 如果是Dataset对象，直接获取text
                texts = dataset["text"]
                
            print(f"[进程{worker_id}|PID:{pid}] ✅ load_dataset加载完成，耗时: {time.time()-dataset_start:.2f}秒")
            
        except Exception as e:
            print(f"[进程{worker_id}|PID:{pid}] ❌ load_dataset失败: {e}")
            # 回退到逐个文件读取
            print(f"[进程{worker_id}|PID:{pid}] 🔄 回退到逐个文件读取...")
            return ExperimentPreparer._process_shard_without_load_dataset(
                worker_id, shard_dir, tokenizer, batch_size, save_path
            )
        
        print(f"[进程{worker_id}|PID:{pid}] 📊 分片包含 {len(texts)} 个文本")
        
        # 批量tokenize
        all_tokens = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        tokenize_start = time.time()
        for batch_idx in range(0, len(texts), batch_size):
            batch_end = min(batch_idx + batch_size, len(texts))
            text_batch = texts[batch_idx:batch_end]
            
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
            
            # 每10个批次打印进度
            if ((batch_idx // batch_size) + 1) % 10 == 0:
                progress = (batch_idx + len(text_batch)) / len(texts) * 100
                elapsed = time.time() - tokenize_start
                speed = (batch_idx + len(text_batch)) / elapsed if elapsed > 0 else 0
                print(f"[进程{worker_id}|PID:{pid}] 📊 进度: {progress:.1f}%，"
                      f"速度: {speed:.1f} 文本/秒")
        
        # 子进程自己保存结果
        print(f"[进程{worker_id}|PID:{pid}] 💾 保存结果到: {save_path}")
        torch.save(all_tokens, save_path)
        
        total_time = time.time() - start_time
        print(f"[进程{worker_id}|PID:{pid}] 🎉 分片处理完成: {len(texts)} 个文本, "
              f"{len(all_tokens)} 个tokens, 总耗时: {total_time:.2f}秒")
        
        return worker_id, len(texts), len(all_tokens), save_path
    
    @staticmethod
    def _process_shard_without_load_dataset(worker_id, shard_dir, tokenizer, batch_size, save_path):
        """回退方案：不使用load_dataset，直接读取文件"""
        import glob
        import time
        
        pid = os.getpid()
        txt_files = glob.glob(os.path.join(shard_dir, "*.txt"))
        print(f"[进程{worker_id}|PID:{pid}] 📁 直接读取 {len(txt_files)} 个文件")
        
        all_tokens = []
        total_files_processed = 0
        
        start_time = time.time()
        
        for batch_idx in range(0, len(txt_files), batch_size):
            batch_files = txt_files[batch_idx:batch_idx + batch_size]
            
            batch_texts = []
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        batch_texts.append(text)
                except Exception as e:
                    print(f"[进程{worker_id}|PID:{pid}] ❌ 读取文件失败 {file_path}: {e}")
                    continue
            
            if not batch_texts:
                continue
            
            # Tokenize
            encoded = tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            
            for seq in encoded:
                all_tokens.extend(seq)
            
            total_files_processed += len(batch_files)
            
            # 每10个批次打印进度
            if ((batch_idx // batch_size) + 1) % 10 == 0:
                progress = total_files_processed / len(txt_files) * 100
                elapsed = time.time() - start_time
                speed = total_files_processed / elapsed if elapsed > 0 else 0
                print(f"[进程{worker_id}|PID:{pid}] 📊 进度: {progress:.1f}%，"
                      f"速度: {speed:.1f} 文件/秒")
        
        # 子进程自己保存结果
        print(f"[进程{worker_id}|PID:{pid}] 💾 保存结果到: {save_path}")
        torch.save(all_tokens, save_path)
        
        total_time = time.time() - start_time
        print(f"[进程{worker_id}|PID:{pid}] 🎉 直接读取完成: {total_files_processed} 个文件, "
              f"{len(all_tokens)} 个tokens, 总耗时: {total_time:.2f}秒")
        
        return worker_id, total_files_processed, len(all_tokens), save_path
    
    def _tokenize_sharded(self):
        """处理分片目录的多进程方法，子进程自己保存文件"""
        print(f"🚀 启动分片目录多进程处理，使用 {self.num_workers} 个进程")
        
        if not self.shard_dirs:
            raise ValueError("分片目录列表为空")
        
        # 创建临时目录用于存储各worker的中间结果
        partial_dir = f"{self.output_file}_partial"
        os.makedirs(partial_dir, exist_ok=True)
        print(f"📂 临时目录: {partial_dir}")
        
        # 准备worker输入数据，包括保存路径
        worker_inputs = []
        for i, shard_dir in enumerate(self.shard_dirs):
            if i >= self.num_workers:  # 限制进程数
                break
            save_path = os.path.join(partial_dir, f"worker_{i}.pt")
            worker_inputs.append((
                i,  # worker_id
                shard_dir,  # 分片目录
                self.tokenizer_name,
                self.model_cache_dir,
                self.batch_size,
                save_path  # 保存路径
            ))
        
        print(f"📊 将处理 {len(worker_inputs)} 个分片")
        
        # 并行处理
        all_tokens = []
        total_files = 0
        total_tokens = 0
        
        with mp.Pool(min(self.num_workers, len(worker_inputs))) as pool:
            results = []
            for result in pool.imap_unordered(
                self._tokenize_shard_worker, worker_inputs
            ):
                worker_id, file_count, token_count, save_path = result
                results.append((worker_id, file_count, token_count, save_path))
                print(f"✅ Worker {worker_id} 完成，处理了 {file_count} 个文件，"
                      f"生成 {token_count} 个tokens")
        
        # 主进程只负责合并
        print(f"🔄 开始合并 {len(results)} 个worker的结果...")
        
        # 按worker_id排序以确保顺序一致
        results.sort(key=lambda x: x[0])
        
        for worker_id, file_count, token_count, save_path in results:
            try:
                # 加载worker保存的结果
                worker_tokens = torch.load(save_path, weights_only=True)
                all_tokens.extend(worker_tokens)
                total_files += file_count
                total_tokens += token_count
                
                # 删除worker的临时文件
                os.remove(save_path)
                print(f"📥 已合并 Worker {worker_id} 的结果")
                
            except Exception as e:
                print(f"❌ 加载Worker {worker_id} 的结果失败: {e}")
        
        # 保存最终结果
        print(f"💾 保存最终结果到: {self.output_file}")
        torch.save(all_tokens, self.output_file)
        
        # 清理临时目录
        try:
            os.rmdir(partial_dir)
            print(f"🗑️  清理临时目录: {partial_dir}")
        except OSError:
            pass
        
        print(f"📊 所有分片处理完成: 总共 {total_files} 个文件, {total_tokens} 个tokens")
        return all_tokens
    
    @staticmethod
    def _tokenize_worker(worker_data):
        """原有的tokenize worker函数"""
        worker_id, text_batches, tokenizer_name, cache_dir = worker_data
        try:
            # 首先尝试从缓存加载
            tokenizer = Qwen2Tokenizer.from_pretrained(
                tokenizer_name, 
                cache_dir=cache_dir,
                local_files_only=True  # 只从本地加载
            )
        except (TypeError, OSError):
            # 如果缓存中没有，则下载（在跳板机上运行）
            print(f"⚠️  缓存中未找到 {tokenizer_name}，开始下载...")
            tokenizer = Qwen2Tokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=cache_dir  # 下载到缓存目录
            )
            print(f"✅ 已下载到缓存: {cache_dir}")

        all_tokens = []
        total_files_processed = 0
        
        for batch_idx, text_batch in enumerate(text_batches):
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
            
            # 每处理完一批就打印进度
            files_in_this_batch = len(text_batch)
            total_files_processed += files_in_this_batch
            
            # 每10个批次或最后一批打印一次
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(text_batches):
                print(f"进程 {worker_id}: 批次 {batch_idx+1}/{len(text_batches)} "
                      f"处理了 {files_in_this_batch} 个文件，"
                      f"累计 {total_files_processed} 个文件")
        
        print(f"✅ 进程 {worker_id} 完成: 总共处理了 {total_files_processed} 个文件")
        return worker_id, all_tokens, len(text_batches), total_files_processed
    
    @staticmethod
    def _distribute_data(texts, num_workers, batch_size):
        """分配数据给各个worker"""
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
        """合并并保存结果"""
        all_tokens = []
        for wid in range(num_workers):
            f = os.path.join(partial_dir, f"worker_{wid}.pt")
            if os.path.exists(f):
                data = torch.load(f, weights_only=True)
                all_tokens.extend(data["tokens"])
        torch.save(all_tokens, output_file)
        return all_tokens
    
    def _tokenize_multi_process(self):
        """多进程处理文本列表"""
        partial_dir = f"{self.output_file}_partial"
        os.makedirs(partial_dir, exist_ok=True)
        logger.info("Using multi-process tokenization start")
        
        # 1. 数据划分
        worker_data = self._distribute_data(self.texts, self.num_workers, self.batch_size)
        worker_inputs = [(wid, batches, self.tokenizer_name, self.model_cache_dir)
                        for wid, batches in worker_data]

        # 2. 并行 tokenize
        results = []
        with mp.Pool(self.num_workers) as pool:
            for wid, tokens, batch_count, file_count in pool.imap_unordered(
                self._tokenize_worker, worker_inputs
            ):
                torch.save({"tokens": tokens}, os.path.join(partial_dir, f"worker_{wid}.pt"))
                results.append((wid, len(tokens), file_count))
                print(f"✅ 进程 {wid} 完成: 处理了 {batch_count} 个批次，"
                      f"共 {file_count} 个文件，生成 {len(tokens)} 个token")

        # 3. 合并 & 清理
        all_tokens = self._merge_and_save(partial_dir, self.num_workers, self.output_file)
        for wid, _, _ in results:
            try:
                os.remove(os.path.join(partial_dir, f"worker_{wid}.pt"))
            except FileNotFoundError:
                pass
        os.rmdir(partial_dir)
        
        # 打印汇总信息
        total_files = sum(file_count for _, _, file_count in results)
        logger.info(f"多进程tokenize完成 -> 总文件数: {total_files}, 总token数: {len(all_tokens)}")
        return all_tokens
    
    def _tokenize_single_process(self):
        """单进程处理文本列表"""
        logger.info("Using single-process tokenization")
        
        # 修正：将 texts 分成批次，与多进程模式保持一致
        batches = [self.texts[i:i + self.batch_size] for i in range(0, len(self.texts), self.batch_size)]
        worker_input = (0, batches, self.tokenizer_name, self.model_cache_dir)
        
        # 直接调用 worker 函数
        _, all_tokens, batch_count, file_count = self._tokenize_worker(worker_input)
        
        # 保存结果
        torch.save(all_tokens, self.output_file)
        logger.info(f"单进程tokenize完成 -> 处理了 {batch_count} 个批次，"
                    f"共 {file_count} 个文件，生成 {len(all_tokens)} 个token")
        return all_tokens
    
    # ==================== 统一的tokenize入口 ====================
    
    def tokenize(self):
        """统一的tokenize方法，根据模式选择处理方式"""
        if os.path.exists(self.output_file):
            logger.info(f"Cache hit -> {self.output_file}")
            return torch.load(self.output_file, weights_only=True)
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # 根据模式选择处理方法
        if self.mode == "sharded":
            return self._tokenize_sharded()
        else:
            # 原有的文本列表处理方式
            if self.num_workers > 1:
                return self._tokenize_multi_process()
            else:
                return self._tokenize_single_process()

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