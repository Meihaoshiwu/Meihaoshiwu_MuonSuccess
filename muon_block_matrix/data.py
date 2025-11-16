import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import random
import time

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import Qwen2Tokenizer
from loguru import logger
from typing import List, Dict, Any, Optional

from .share_mem_manager import StreamConfig, SharedBufferManager
from .config import TOKENIZED_CACHE, OPENWEBTEXT_EXTRACTED, DATASET_CACHE, MODEL_CACHE
#=================================================================

class DataLoaderProcess:
    """数据加载进程 - 负责流式读取文件、tokenize、写入共享缓冲区"""
    def __init__(self, process_id: int, stream_config: StreamConfig, shared_info: Dict[str, Any],
                 dataset_name: str, tokenizer_name: str, file_list: List[str]):
        self.process_id = process_id
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.file_list = file_list
        self.running = True
        self.stream_config = stream_config
        self.shared_info = shared_info
        
        # 静态文件分配 - 避免重叠
        self.assigned_files = self._assign_files_statically()
        self.current_file_index = 0
        self.epoch_count = 0
        self.total_samples_processed = 0
        
        # 初始化tokenizer
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            tokenizer_name, 
            cache_dir=MODEL_CACHE,
            local_files_only=True
        )
        
        logger.info(f"Loader {process_id}: 初始化完成, 分配文件数: {len(self.assigned_files)}")
    
    def _assign_files_statically(self) -> List[str]:
        """静态分配文件，确保不重叠"""
        # 打乱文件确保随机性
        shuffled_files = random.sample(self.file_list, len(self.file_list))
        
        # 按进程数量分配
        files_per_loader = len(shuffled_files) // self.stream_config.num_loaders
        start_idx = self.process_id * files_per_loader
        end_idx = start_idx + files_per_loader
        
        # 最后一个进程处理剩余文件
        if self.process_id == self.stream_config.num_loaders - 1:
            end_idx = len(shuffled_files)
            
        assigned = shuffled_files[start_idx:end_idx]
        logger.info(f"Loader {self.process_id}: 分配文件范围 {start_idx}-{end_idx}, 实际文件数: {len(assigned)}")
        return assigned
    
    def run(self):
        """主运行循环 - 无限数据流"""
        logger.info(f"Loader {self.process_id}: 启动无限数据流")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            self.buffer_manager = SharedBufferManager(self.stream_config, self.shared_info)
            while self.running and consecutive_failures < max_consecutive_failures:
                # 获取可写缓冲区
                buffer_id = self.buffer_manager.get_available_write_buffer()
                if buffer_id is None:
                    consecutive_failures += 1
                    if consecutive_failures % 5 == 0:  # 每5次失败记录一次警告
                        logger.warning(f"Loader {self.process_id}: 第 {consecutive_failures} 次获取缓冲区失败")
                    time.sleep(1.0)
                    continue
                
                consecutive_failures = 0  # 重置失败计数
                
                try:
                    # 收集一个chunk的样本
                    samples = self._collect_samples_for_chunk()
                    if not samples:
                        logger.warning(f"Loader {self.process_id}: 无法收集到样本，开始新epoch")
                        self._start_new_epoch()
                        continue
                    
                    # Tokenize并写入
                    logger.info(f"Loader {self.process_id}: Tokenize {len(samples)}样本")
                    token_chunk = self._tokenize_samples(samples)
                    
                    if token_chunk is not None:
                        self.buffer_manager.write_to_buffer(buffer_id, token_chunk)
                        self.total_samples_processed += len(samples)
                        
                        # 输出缓冲区状态
                        stats = self.buffer_manager.get_buffer_stats()
                        logger.info(f"Loader {self.process_id}: 已处理 {self.total_samples_processed} 样本, "
                                f"缓冲区状态 - 空:{stats['empty']} 写:{stats['writing']} "
                                f"满:{stats['full']} 读:{stats['reading']} 满比例:{stats['full_ratio']:.1%}")
                    else:
                        logger.error(f"Loader {self.process_id}: Tokenize失败")
                        self.buffer_manager.release_write_buffer(buffer_id)
                    
                except Exception as e:
                    logger.error(f"Loader {self.process_id}: 处理错误 - {e}")
                    # 确保在异常时释放缓冲区
                    try:
                        self.buffer_manager.release_write_buffer(buffer_id)
                    except:
                        pass
        finally: 
            self.buffer_manager.close()   
            logger.info(f"Loader {self.process_id}: 退出, 总共处理 {self.total_samples_processed} 样本")
    
    def _collect_samples_for_chunk(self) -> List[str]:
        """收集一个chunk所需的样本 - 确保数量正确"""
        samples_needed = self.buffer_manager.config.batches_per_chunk * self.buffer_manager.config.batch_size
        collected_samples = []
        
        max_attempts = 10  # 最大尝试次数，避免无限循环
        attempts = 0
        
        while len(collected_samples) < samples_needed and self.running and attempts < max_attempts:
            attempts += 1
            
            # 检查是否需要开始新epoch
            if self.current_file_index >= len(self.assigned_files):
                self._start_new_epoch()
            
            # 从当前文件读取样本
            current_file = self.assigned_files[self.current_file_index]
            new_samples = self._read_samples_from_file(current_file, samples_needed - len(collected_samples))
            
            if not new_samples:
                # 当前文件已读完或读取失败，移动到下一个文件
                self.current_file_index += 1
                logger.debug(f"Loader {self.process_id}: 完成文件 {current_file}, 移动到下一个文件")
                continue
            
            collected_samples.extend(new_samples)
        
        # 如果样本不足，丢弃当前收集的样本
        if len(collected_samples) < samples_needed:
            logger.warning(f"Loader {self.process_id}: 样本不足 {len(collected_samples)} < {samples_needed}，丢弃当前批次")
            return []
        
        # 打乱样本顺序
        random.shuffle(collected_samples)
        logger.debug(f"Loader {self.process_id}: 收集并打乱 {len(collected_samples)} 个样本")
        
        return collected_samples
    
    def _start_new_epoch(self):
        """开始新的epoch，重新打乱文件顺序，这里是静态分配的，打乱的是自己的文件，避免还需要进程通信"""
        self.epoch_count += 1
        self.current_file_index = 0
        
        # 重新打乱文件顺序
        random.shuffle(self.assigned_files)
        
        logger.info(f"Loader {self.process_id}: 开始第 {self.epoch_count} 个epoch, 文件数: {len(self.assigned_files)}")
    
    def _read_samples_from_file(self, file_path: str, max_samples: int) -> List[str]:
        """从文件读取样本"""
        samples = []
        
        try:
            if not os.path.exists(file_path):
                logger.error(f"Loader {self.process_id}: 文件不存在 {file_path}")
                return []
            
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(samples) >= max_samples:
                            break
                        text = line.strip()
                        if text:  # 跳过空行
                            samples.append(text)
            elif file_path.endswith('.jsonl'):
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(samples) >= max_samples:
                            break
                        try:
                            data = json.loads(line)
                            text = data.get('text', '')
                            if text:
                                samples.append(text)
                        except json.JSONDecodeError:
                            continue
            else:
                logger.warning(f"Loader {self.process_id}: 不支持的文件格式 {file_path}")
                
        except Exception as e:
            logger.error(f"Loader {self.process_id}: 读取文件 {file_path} 错误 - {e}")
        
        logger.debug(f"Loader {self.process_id}: 从 {file_path} 读取 {len(samples)} 样本")
        return samples
    
    def _tokenize_samples(self, samples: List[str]) -> Optional[torch.Tensor]:
        """Tokenize样本并组装成batch - 添加格式验证"""
        if not samples:
            return None
            
        batch_size = self.buffer_manager.config.batch_size
        max_length = self.buffer_manager.config.max_length
        batches_per_chunk = self.buffer_manager.config.batches_per_chunk
        
        try:
            # 验证输入样本数量
            total_samples_needed = batches_per_chunk * batch_size
            assert len(samples) == total_samples_needed, \
                f"样本数量不正确: {len(samples)} != {total_samples_needed}"
            
            # 批量tokenize
            encoded = self.tokenizer.batch_encode_plus(
                samples,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"]  # [total_samples, max_length]
            
            # 验证tokenize后的形状
            assert input_ids.size(0) == total_samples_needed, \
                f"Tokenize后样本数量变化: {input_ids.size(0)} != {total_samples_needed}"
            assert input_ids.size(1) == max_length, \
                f"序列长度不正确: {input_ids.size(1)} != {max_length}"
            
            # 重新组织为 [batches_per_chunk, batch_size, max_length]
            reshaped_data = input_ids.view(batches_per_chunk, batch_size, max_length)
            
            logger.debug(f"Loader {self.process_id}: Tokenize完成，形状 {tuple(reshaped_data.shape)}")
            return reshaped_data
            
        except Exception as e:
            logger.error(f"Loader {self.process_id}: Tokenize错误 - {e}")
            return None
    
    def stop(self):
        """停止加载进程"""
        self.running = False
        logger.info(f"Loader {self.process_id}: 停止信号接收")

# 工具函数
def start_data_loaders(dataset_name: str,  stream_config: StreamConfig, shared_info: Dict[str, Any],
                      file_list: List[str], tokenizer_name: str = "Qwen/Qwen2.5-0.5B") -> List:
    """启动数据加载进程"""
    processes = []
    for i in range(stream_config.num_loaders):
        loader = DataLoaderProcess(i, stream_config, shared_info, dataset_name, tokenizer_name, file_list)
        p = torch.multiprocessing.Process(target=loader.run, daemon=True)
        p.start()
        processes.append((p, loader))
    
    logger.info(f"启动 {stream_config.num_loaders} 个数据加载进程")
    return processes

def stop_data_loaders(loader_processes: List):
    """停止数据加载进程"""
    logger.info("停止数据加载进程")
    for p, loader in loader_processes:
        loader.stop()
        p.join(timeout=5.0)
        if p.is_alive():
            logger.warning(f"进程 {p.pid} 仍在运行，强制终止")
            p.terminate()
    
    logger.info("数据加载进程已停止")

def get_all_data_files(data_dir: str, extensions: List[str] = None) -> List[str]:
    """获取所有数据文件"""
    if extensions is None:
        extensions = ['.txt', '.jsonl']
    
    file_list = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))
    
    logger.info(f"找到 {len(file_list)} 个数据文件在目录 {data_dir}")
    return file_list

def load_dataset_from_files(dataset_name: str) -> List[str]:
    """根据数据集名称获取文件列表"""
    # 这里根据您的实际数据集配置实现
    # 示例：从配置或固定路径获取
    if dataset_name == "openwebtext":
        data_dir = "/path/to/openwebtext/files"
    elif dataset_name == "wikitext":
        data_dir = "/path/to/wikitext/files"
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    return get_all_data_files(data_dir)

class StreamingMoonDataset:
    """流式MoonDataset - 从共享缓冲区读取数据，支持DDP同步"""
    
    def __init__(
        self,
        buffer_manager,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = None
    ):
        self.config = buffer_manager.config
        if self.config.batch_size % world_size != 0:
            raise ValueError(f"batch_size({self.config.batch_size})必须能被world_size({world_size})整除")
        self.my_batch_count = self.config.batches_per_chunk # 每个chunk存放多少batch是固定下来的
        self.buffer_manager = buffer_manager
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        # 当前chunk状态
        self.current_buffer_id = None
        self.current_chunk_data = None
        self.current_batch_index = 0
        
        # 统计信息
        self.total_batches_processed = 0
        self.total_tokens_processed = 0
        
        logger.info(f"Rank {rank}: 初始化流式数据集, device: {self.device}")
    
    def __getitem__(self, idx):
        """获取单个样本 - 兼容原有接口"""
        batch = self.get_next_batch()
        if batch is None:
            raise StopIteration("没有更多数据")
        return batch
    
    def get_next_batch(self):
        """获取下一个batch - 包含DDP同步"""
        # 如果当前chunk为空或已处理完，需要获取新chunk
        if self.current_chunk_data is None or self.current_batch_index >= self.my_batch_count:
            # 到这里当前chunk必然已经用完了
            self._release_current_chunk_sync()
            success = self._acquire_new_chunk_sync()
            if not success:
                logger.warning(f"Rank {self.rank}: 无法获取新chunk")
                return None
        
        # 获取当前rank对应的batch
        batch = self._get_batch_for_rank(self.current_batch_index)
        self.current_batch_index += 1
        self.total_batches_processed += 1
        self.total_tokens_processed += self.config.batch_size * self.config.max_length
        
        # 直接传输到GPU
        if batch is not None:
            batch = batch.to(self.device, non_blocking=True).long()
            logger.debug(f"Rank {self.rank}: 获取batch {self.current_batch_index-1}/{self.my_batch_count}, "
                        f"形状: {batch.shape}")
        else:
            logger.warning(f"Rank {self.rank}: 获取到空batch")
        
        return batch
    
    def _acquire_new_chunk_sync(self) -> bool:
        """同步获取新chunk - rank 0负责协调"""
        # 确保DDP已初始化
        if self.world_size > 1 and (not dist.is_initialized()):
            logger.error(f"Rank {self.rank}: DDP未初始化")
            return False
        if self.rank == 0:
            # 主进程负责获取新chunk
            # 重试机制：等待生产者提供数据
            max_retries = 60  # 最大重试次数（比如60秒）
            retry_count = 0
            
            while retry_count < max_retries:
                buffer_id = self.buffer_manager.get_available_read_buffer()
                if buffer_id is not None:  # 修复：检查是否为None
                    break
                    
                # 获取失败，等待后重试
                retry_count += 1
                if retry_count % 10 == 0:  # 每10次重试记录一次日志
                    logger.warning(f"Rank {self.rank}: 第 {retry_count} 次尝试获取chunk失败，继续等待...")
                
                time.sleep(1.0)  # 等待1秒
        
            if buffer_id is None:  # 修复：检查是否为None
                logger.error(f"Rank {self.rank}: 经过 {max_retries} 次尝试仍无法获取chunk，训练可能停止")
                raise RuntimeError("无法从数据加载器获取数据，训练停止")
                
            if self.world_size > 1:
                # 广播buffer_id给所有进程
                buffer_id_tensor = torch.tensor([buffer_id], dtype=torch.int, device=self.device)
                dist.broadcast(buffer_id_tensor, src=0)
                logger.info(f"Rank 0: 获取chunk {buffer_id} 并广播")
        else:
            # 其他进程等待广播
            buffer_id_tensor = torch.tensor([0], dtype=torch.int, device=self.device)
            dist.broadcast(buffer_id_tensor, src=0)
            buffer_id = buffer_id_tensor.item()

        # 到这一步必然获取到合法的chunkID，从共享内存读取本训练进程的数据
        self.current_buffer_id = buffer_id
        chunk_data = self.buffer_manager.read_from_buffer(buffer_id, self.rank)
        if chunk_data is None:
            self.buffer_manager.release_read_buffer(buffer_id)
            logger.error(f"Rank {self.rank}: 接收chunk {buffer_id} is None!")
            raise RuntimeError(f"Rank {self.rank}:共享内存获取数据异常")
        assert chunk_data.shape[0] == self.config.batches_per_chunk, \
            f"数据形状不一致: {chunk_data.shape[0]} != {self.config.batches_per_chunk}"
        self.current_chunk_data = chunk_data
        self.current_batch_index = 0
        logger.info(f"Rank {self.rank}: 接收chunk {buffer_id}, 形状: {self.current_chunk_data.shape}")
        return True
    
    def _get_batch_for_rank(self, batch_index: int) -> Optional[torch.Tensor]:
        """根据rank获取对应的batch数据"""
        if self.current_chunk_data is None:
            return None
        
        if batch_index < self.config.batches_per_chunk:
            return self.current_chunk_data[batch_index]
        
        return None
    
    def _release_current_chunk_sync(self):
        """同步释放当前chunk"""
        if self.current_buffer_id is not None:
            if self.rank == 0:
                self.buffer_manager.release_read_buffer(self.current_buffer_id)
                logger.info(f"Rank 0: 释放chunk {self.current_buffer_id}")
        
        self.current_buffer_id = None
        self.current_chunk_data = None
        self.current_batch_index = 0
        self.my_batch_count = 0
        
        logger.debug(f"Rank {self.rank}: 重置chunk状态")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'total_batches_processed': self.total_batches_processed,
            'total_tokens_processed': self.total_tokens_processed,
            'current_buffer_id': self.current_buffer_id,
            'current_batch_index': self.current_batch_index,
            'my_batch_count': self.my_batch_count
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_batches_processed = 0
        self.total_tokens_processed = 0
#=================================================================

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
        model_cache_dir: str,
        output_file: str,
        batch_size: int = 500,
    ):
        self.texts = texts
        self.tokenizer_name = tokenizer_name
        self.model_cache_dir = model_cache_dir
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
        except (TypeError, OSError):
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
        worker_inputs = [(wid, batches, self.tokenizer_name, self.model_cache_dir)
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
        worker_input = (0, batches, self.tokenizer_name, self.model_cache_dir)
        
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