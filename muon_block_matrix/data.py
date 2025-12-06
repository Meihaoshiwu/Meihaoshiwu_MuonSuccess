import os
import torch
import torch.multiprocessing as mp
import glob
import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import Qwen2Tokenizer
from loguru import logger
from typing import List, Optional

from .config import TOKENIZED_CACHE, DATASET_CACHE

def load_dataset_by_name(dataset_name: str):
    """加载数据集函数"""
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
        "openwebtext": "Skylion007/openwebtext",
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
        num_workers,
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
        self.num_workers = num_workers
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
        import traceback
        from loguru import logger
        
        pid = os.getpid()
        log_file = f"logs/worker_{worker_id}_pid_{pid}.log"
        os.makedirs("logs", exist_ok=True)
        logger.add(log_file, rotation="10 MB", retention=3)
        
        # 记录详细错误信息到文件
        error_log_file = f"logs/worker_{worker_id}_errors.log"
        
        try:
            logger.info(f"[进程{worker_id}|PID:{pid}] 🔄 开始处理分片: {shard_dir}")
            
            # 记录任务开始时间
            task_start_time = time.time()
            
            # 加载tokenizer
            start_time = time.time()
            try:
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    tokenizer_name, 
                    cache_dir=cache_dir,
                    local_files_only=True
                )
            except (TypeError, OSError):
                logger.warning(f"缓存中未找到 {tokenizer_name}，开始下载...")
                tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
            
            logger.info(f"📂 加载分片文件...")
            
            try:
                txt_files = glob.glob(f"{shard_dir}/*.txt")
                logger.info(f"找到 {len(txt_files)} 个文本文件")

                texts = []

                for file_path in txt_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            # 使用列表推导式一次性读取并处理
                            file_texts = [line.strip() for line in f if line.strip()]
                            texts.extend(file_texts)
                    except Exception as e:
                        logger.error(f"读取文件 {file_path} 时出错: {e}")
                        # 记录到错误文件
                        with open(error_log_file, "a") as err_f:
                            err_f.write(f"文件读取错误 {file_path}: {e}\n")
                
            except Exception as e:
                error_msg = f"从目录{shard_dir}获取文件失败: {e}"
                logger.error(error_msg)
                with open(error_log_file, "a") as err_f:
                    err_f.write(f"{error_msg}\n{traceback.format_exc()}\n")
                raise
            
            logger.info(f"📊 分片包含 {len(texts)} 个文本")
            
            # 批量tokenize
            all_tokens = []

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

                if (batch_idx) % (1000000) == 0:
                    logger.info(f"批次 {batch_idx}")
            
            # 保存结果
            torch.save(all_tokens, save_path)
            
            total_time = time.time() - start_time
            task_total_time = time.time() - task_start_time
            logger.info(f"🎉 分片处理完成: {len(texts)} 个文本, "
                        f"{len(all_tokens)} 个tokens, tokenize耗时: {total_time:.2f}秒, "
                        f"总耗时: {task_total_time:.2f}秒, 保存结果到: {save_path}")
            
            return worker_id, len(texts), len(all_tokens), save_path
            
        except Exception as e:
            # 记录详细错误信息
            error_msg = f"Worker {worker_id} (PID: {pid}) 失败: {e}"
            full_traceback = traceback.format_exc()
            
            logger.critical(error_msg)
            logger.critical(full_traceback)
            
            # 写入错误文件
            with open(error_log_file, "a") as err_f:
                err_f.write(f"任务失败时间: {time.ctime()}\n")
                err_f.write(f"Worker ID: {worker_id}, PID: {pid}\n")
                err_f.write(f"分片目录: {shard_dir}\n")
                err_f.write(f"错误信息: {error_msg}\n")
                err_f.write(f"完整堆栈:\n{full_traceback}\n")
                err_f.write("="*80 + "\n")
            
            # 重新抛出异常，让父进程知道
            raise
    
    def _tokenize_sharded(self):
        """处理分片目录的多进程方法，支持失败重试"""
        print(f"🚀 启动分片目录多进程处理，使用 {self.num_workers} 个进程")
        
        if not self.shard_dirs:
            raise ValueError("分片目录列表为空")
        
        # 创建临时目录用于存储各worker的中间结果
        partial_dir = f"{self.output_file}_partial"
        os.makedirs(partial_dir, exist_ok=True)
        print(f"📂 临时目录: {partial_dir}")
        print(f"📂 分片数量: {len(self.shard_dirs)}，子进程数量:{self.num_workers}")
        
        # 准备初始任务
        initial_tasks = []
        for i, shard_dir in enumerate(self.shard_dirs):
            if i >= self.num_workers:  # 限制进程数
                break
            save_path = os.path.join(partial_dir, f"worker_{i}.pt")
            initial_tasks.append((
                i,  # worker_id
                shard_dir,  # 分片目录
                self.tokenizer_name,
                self.model_cache_dir,
                self.batch_size,
                save_path  # 保存路径
            ))
        
        print(f"📊 将处理 {len(initial_tasks)} 个目录")
        
        # 重试逻辑
        max_retries = 2  # 最大重试次数
        all_results = []  # 成功的结果
        failed_tasks = []  # 失败的任务
        
        # 第一轮处理
        remaining_tasks = initial_tasks.copy()
        
        for retry_round in range(max_retries + 1):  # +1 因为包含初始运行
            if not remaining_tasks:
                break
                
            if retry_round > 0:
                print(f"\n🔄 开始第 {retry_round} 轮重试，剩余 {len(remaining_tasks)} 个任务")
            
            # 本轮成功和失败的任务
            round_success = []
            round_failed = []
            
            # 处理当前剩余任务
            with mp.Pool(min(self.num_workers, len(remaining_tasks))) as pool:
                async_results = []
                
                # 提交所有任务
                for task in remaining_tasks:
                    worker_id = task[0]
                    async_result = pool.apply_async(
                        self._tokenize_shard_worker,
                        args=(task,),
                        error_callback=lambda e, wid=worker_id: 
                            print(f"❌ Worker {wid} 异步回调报告异常: {e}")
                    )
                    async_results.append((worker_id, async_result))
                
                # 收集结果（带超时）
                for worker_id, async_result in async_results:
                    try:
                        # 设置超时（例如6小时）
                        result = async_result.get(timeout=3600)
                        round_success.append(result)
                        print(f"✅ Worker {worker_id} 完成，处理了 {result[1]} 个文件，"
                            f"生成 {result[2]} 个tokens")
                        
                    except mp.TimeoutError:
                        print(f"⏰ Worker {worker_id} 超时（6小时），将重试")
                        # 找出对应的任务
                        failed_task = next(t for t in remaining_tasks if t[0] == worker_id)
                        round_failed.append(failed_task)
                        
                    except Exception as e:
                        print(f"❌ Worker {worker_id} 失败: {e}")
                        import traceback
                        traceback.print_exc()
                        # 找出对应的任务
                        failed_task = next(t for t in remaining_tasks if t[0] == worker_id)
                        round_failed.append(failed_task)
            
            # 更新结果
            all_results.extend(round_success)
            remaining_tasks = round_failed
            
            # 如果本轮没有失败，退出循环
            if not remaining_tasks:
                print(f"\n🎉 第 {retry_round} 轮后所有任务成功完成")
                break
        
        # 最终报告
        print(f"\n📊 最终结果: 成功 {len(all_results)}/{len(initial_tasks)} 个任务")
        if remaining_tasks:
            failed_ids = [task[0] for task in remaining_tasks]
            print(f"❌ 以下任务失败（尝试 {max_retries+1} 次后）: {failed_ids}")
        
        # 合并所有成功的结果
        all_tokens = []
        total_files = 0
        total_tokens = 0
        
        # 按worker_id排序以确保顺序一致
        all_results.sort(key=lambda x: x[0])
        
        print(f"🔄 开始合并 {len(all_results)} 个worker的结果...")
        
        for worker_id, file_count, token_count, save_path in all_results:
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
            # 检查是否还有残留文件
            remaining_files = os.listdir(partial_dir)
            if remaining_files:
                print(f"⚠️  临时目录中仍有文件: {remaining_files}")
            else:
                os.rmdir(partial_dir)
                print(f"🗑️  清理临时目录: {partial_dir}")
        except OSError:
            pass
        
        print(f"📊 处理完成: 总共 {total_files} 个文件, {total_tokens} 个tokens")
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
            
            # 每1000个批次或最后一批打印一次
            if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == len(text_batches):
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
        logger.info(f"已加载{dataset_name}.bin")
        logger.info(f"加载预处理数据: {len(self.tokens)} tokens")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        return torch.tensor(self.tokens[start:end], dtype=torch.long)
    
class MMapDataset(Dataset):
    def __init__(self, file_path: str, max_length: int = 1024):
        self.file_path = file_path
        self.max_length = max_length
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenized文件不存在: {file_path}")
        
        # 关键：内存映射，零内存压力
        # 注意：假设文件存储的是uint16的token ID（LLaMA-7B和Qwen2.5-7B都适用）
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        self.total_sequences = self.total_tokens // self.max_length
        
        logger.info(f"📁 内存映射: {os.path.basename(file_path)}")
        logger.info(f"  总Tokens: {self.total_tokens:,}, 序列长度: {max_length}")
        logger.info(f"  总样本数: {self.total_sequences:,}")
    
    def __len__(self):
        """返回总样本数（序列数）"""
        return self.total_sequences
    
    def __getitem__(self, idx):
        """根据索引返回一个序列"""
        start = idx * self.max_length
        end = start + self.max_length
        
        # 从内存映射中读取（按需加载）
        sequence = self.data[start:end]
        
        # 转换为torch张量，与原有接口完全一致
        return torch.from_numpy(sequence.astype(np.int64))
    
    def __repr__(self):
        return f"MMapDataset(file={os.path.basename(self.file_path)}, sequences={self.total_sequences:,})"