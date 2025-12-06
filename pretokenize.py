#!/usr/bin/env python3
"""
LLaMA-7B专用分布式Tokenization处理器
输出格式: uint16二进制文件，可直接内存映射
"""

import os
import glob
import time
import traceback
import numpy as np
from typing import List
from transformers import AutoTokenizer
from loguru import logger
import torch.multiprocessing as mp
from datetime import datetime

# 环境变量路径
MUON_BLOCK_MATRIX_EXPERIMENT_DIR = os.getenv("MUON_BLOCK_MATRIX_EXPERIMENT_DIR")
MODEL_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Models")
RESULTS_BASE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Results")
BASE_LOG_PATH = os.path.join(RESULTS_BASE, "Logs/MuonBlockMatrix")
DATA_EXTRACTED_DIR = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Datasets", "extracted_data", "openwebtext")
TOKENIZED_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "TokenizedData")

class Llama7BTokenizer:
    """LLaMA-7B专用Tokenizer处理器"""
    
    # 固定配置
    MODEL_NAME = "huggyllama/llama-7b"
    VOCAB_SIZE = 32000
    DTYPE = np.uint16  # 固定使用2字节
    
    def __init__(
        self,
        num_workers: int = 32,
        text_data_dir: str = None,
        output_dir: str = None,
        batch_size: int = 1000
    ):
        """
        初始化LLaMA-7B Tokenizer处理器
        
        Args:
            num_workers: 子进程数量，默认32
            text_data_dir: 原始文本数据目录
            output_dir: 输出目录
            batch_size: 每批处理的文本数量
        """
        self.num_workers = num_workers
        self.text_data_dir = text_data_dir
        self.output_dir = output_dir or TOKENIZED_CACHE
        self.batch_size = batch_size
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        logger.info("=" * 60)
        logger.info("LLaMA-7B分布式Tokenization处理器")
        logger.info(f"工作进程数: {self.num_workers}")
        logger.info(f"数据目录: {self.text_data_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"数据类型: {self.DTYPE} (2字节)")
        logger.info("=" * 60)
    
    def _setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(BASE_LOG_PATH, f"llama_tokenization_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        
        # 主日志文件
        log_file = os.path.join(log_dir, "tokenization.log")
        logger.add(
            log_file,
            rotation="100 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        # 错误日志文件
        self.error_log_file = os.path.join(log_dir, "errors.log")
        logger.add(
            self.error_log_file,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}\n{exception}"
        )
        
        self.log_dir = log_dir
        logger.info(f"日志目录: {log_dir}")
    
    def _download_tokenizer(self) -> bool:
        """下载LLaMA-7B Tokenizer"""
        logger.info(f"下载Tokenizer: {self.MODEL_NAME}")
        
        try:
            # 首先尝试从缓存加载
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.MODEL_NAME,
                    cache_dir=MODEL_CACHE,
                    local_files_only=True
                )
                logger.info("✅ Tokenizer已存在于缓存中")
                return True
            except OSError:
                logger.warning("⚠️  缓存中未找到Tokenizer，开始下载...")
            
            # 下载Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                cache_dir=MODEL_CACHE
            )
            
            # 验证下载
            actual_vocab_size = tokenizer.vocab_size
            if actual_vocab_size != self.VOCAB_SIZE:
                logger.warning(f"⚠️  词汇表大小不匹配: 期望 {self.VOCAB_SIZE}, 实际 {actual_vocab_size}")
            
            logger.info(f"✅ Tokenizer下载完成，词汇表大小: {actual_vocab_size:,}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tokenizer下载失败: {e}")
            return False
    
    def _get_text_files(self) -> List[str]:
        """获取所有文本文件路径"""
        try:
            pattern = os.path.join(self.text_data_dir, "**", "*.txt")
            text_files = glob.glob(pattern, recursive=True)
            
            if not text_files:
                raise FileNotFoundError(f"在目录 {self.text_data_dir} 中未找到.txt文件")
            
            logger.info(f"找到 {len(text_files)} 个文本文件")
            return text_files
            
        except Exception as e:
            logger.error(f"获取文本文件失败: {e}")
            raise
    
    def _tokenize_worker(
        self, 
        worker_id: int, 
        file_chunk: List[str], 
        output_file: str
    ):
        """
        单个工作进程的tokenization逻辑
        """
        # 为每个worker设置独立日志
        worker_logger = logger.bind(worker_id=worker_id)
        
        try:
            worker_logger.info(f"启动Worker {worker_id}, 处理 {len(file_chunk)} 个文件")
            
            # 加载tokenizer（强制离线）
            start_time = time.time()
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.MODEL_NAME,
                    cache_dir=MODEL_CACHE,
                    local_files_only=True
                )
                load_time = time.time() - start_time
                worker_logger.info(f"Tokenizer加载成功，耗时: {load_time:.2f}s")
            except Exception as e:
                worker_logger.error(f"Tokenizer加载失败: {e}")
                raise
            
            # 打开输出文件
            with open(output_file, 'wb') as f_bin:
                total_tokens = 0
                files_processed = 0
                
                for file_path in file_chunk:
                    try:
                        # 读取文件
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read().strip()
                        
                        if not text:
                            worker_logger.debug(f"文件为空: {file_path}")
                            continue
                        
                        # Tokenization
                        tokens = tokenizer.encode(text, add_special_tokens=True)
                        
                        if not tokens:
                            worker_logger.debug(f"Tokenization结果为空: {file_path}")
                            continue
                        
                        # 转换为uint16并写入
                        token_array = np.array(tokens, dtype=self.DTYPE)
                        token_array.tofile(f_bin)
                        
                        total_tokens += len(tokens)
                        files_processed += 1
                        
                        # 进度记录
                        if files_processed % 100 == 0:
                            worker_logger.info(
                                f"进度: {files_processed}/{len(file_chunk)} 文件, "
                                f"{total_tokens:,} tokens"
                            )
                            
                    except Exception as e:
                        error_msg = f"处理文件失败 {file_path}: {e}"
                        worker_logger.error(error_msg)
                        # 记录错误但不停止
                        with open(self.error_log_file, 'a') as err_f:
                            err_f.write(f"{datetime.now()}: Worker {worker_id} - {error_msg}\n")
                
                worker_logger.info(
                    f"Worker {worker_id} 完成: "
                    f"处理 {files_processed} 个文件, "
                    f"生成 {total_tokens:,} tokens"
                )
                
                return worker_id, total_tokens, files_processed
                
        except Exception as e:
            error_msg = f"Worker {worker_id} 崩溃: {e}\n{traceback.format_exc()}"
            worker_logger.critical(error_msg)
            with open(self.error_log_file, 'a') as err_f:
                err_f.write(f"\n{datetime.now()}: Worker {worker_id} 崩溃\n")
                err_f.write(traceback.format_exc())
            raise
    
    def tokenize(self, dataset_name: str) -> str:
        """
        执行分布式tokenization
        
        Returns:
            生成的tokenized文件路径
        """
        logger.info(f"开始处理数据集: {dataset_name}")
        start_time = time.time()
        
        try:
            # 1. 下载或验证tokenizer
            if not self._download_tokenizer():
                raise RuntimeError("Tokenizer不可用，无法继续")
            
            # 2. 获取所有文本文件
            text_files = self._get_text_files()
            
            # 3. 将文件均匀分配给各个worker
            chunk_size = len(text_files) // self.num_workers
            remainder = len(text_files) % self.num_workers
            
            file_chunks = []
            start_idx = 0
            
            for i in range(self.num_workers):
                end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
                file_chunks.append(text_files[start_idx:end_idx])
                start_idx = end_idx
            
            logger.info(f"文件分配完成: 每个worker处理 {chunk_size}~{chunk_size+1} 个文件")
            
            # 4. 准备输出文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_file = os.path.join(self.output_dir, f"{dataset_name}_{timestamp}")
            
            # 临时文件
            temp_files = [f"{base_output_file}_worker_{i}.tmp" for i in range(self.num_workers)]
            # 最终输出文件
            final_output_file = f"{base_output_file}.bin"
            
            # 5. 创建进程池并执行
            logger.info(f"启动 {self.num_workers} 个worker进程...")
            
            with mp.Pool(processes=self.num_workers) as pool:
                # 准备worker参数
                worker_args = [
                    (i, file_chunks[i], temp_files[i]) 
                    for i in range(self.num_workers)
                ]
                
                # 异步执行
                async_results = []
                for args in worker_args:
                    result = pool.apply_async(self._tokenize_worker, args=args)
                    async_results.append(result)
                
                # 收集结果
                worker_results = []
                for i, result in enumerate(async_results):
                    try:
                        worker_id, tokens, files = result.get(timeout=7200)  # 2小时超时
                        worker_results.append((worker_id, tokens, files, temp_files[i]))
                        logger.info(f"Worker {worker_id} 完成: {tokens:,} tokens")
                    except mp.TimeoutError:
                        logger.error(f"Worker {i} 超时")
                    except Exception as e:
                        logger.error(f"Worker {i} 失败: {e}")
            
            # 6. 合并所有临时文件
            if not worker_results:
                raise RuntimeError("所有worker都失败了")
            
            logger.info(f"开始合并 {len(worker_results)} 个worker的结果...")
            
            total_tokens = 0
            with open(final_output_file, 'wb') as final_f:
                for worker_id, tokens, files, temp_file in sorted(worker_results):
                    try:
                        # 读取并追加
                        with open(temp_file, 'rb') as temp_f:
                            while chunk := temp_f.read(1024 * 1024):  # 1MB块
                                final_f.write(chunk)
                        
                        total_tokens += tokens
                        logger.info(f"已合并 Worker {worker_id}: {tokens:,} tokens")
                        
                        # 删除临时文件
                        os.remove(temp_file)
                        
                    except Exception as e:
                        logger.error(f"合并Worker {worker_id} 失败: {e}")
            
            # 7. 最终统计
            file_size_gb = os.path.getsize(final_output_file) / (1024 ** 3)
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("🎉 Tokenization 完成!")
            logger.info(f"输出文件: {final_output_file}")
            logger.info(f"文件大小: {file_size_gb:.2f} GB")
            logger.info(f"总Tokens: {total_tokens:,}")
            logger.info(f"总耗时: {total_time:.2f} 秒")
            logger.info(f"处理速度: {total_tokens/total_time:.0f} tokens/秒")
            logger.info("=" * 60)
            
            return final_output_file
            
        except Exception as e:
            error_msg = f"Tokenization流程失败: {e}\n{traceback.format_exc()}"
            logger.critical(error_msg)
            raise RuntimeError(error_msg) from e

def main():
    try:
        # 创建处理器
        processor = Llama7BTokenizer(
            num_workers=32,
            text_data_dir=DATA_EXTRACTED_DIR,
            output_dir=TOKENIZED_CACHE
        )
        
        # 执行tokenization
        output_file = processor.tokenize("openwebtext")
        
        print(f"\n✅ Tokenization完成!")
        print(f"输出文件: {output_file}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    exit(main())