import gc
import time
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup
from loguru import logger
from datetime import datetime
from contextlib import contextmanager
from tqdm import tqdm

from .config import *
from .model import create_qwen_model
from .data import StreamingMoonDataset, MoonDataset, start_data_loaders, stop_data_loaders, load_dataset_from_files
from .optimizer import get_optimizer, STEP_MAP
from .share_mem_manager import SharedMemoryCreator, SharedBufferManager, StreamConfig

# -------------- 工具函数 --------------
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_ddp(rank: int, world_size: int):
    # 增加 NCCL 超时时间
    import os
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时
    os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'
    
    logger.info(f"Rank {rank}: 初始化进程组，超时时间1800秒")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info(f"Rank {rank}: 进程组初始化完成")

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def broadcast_stop_signal(stop_training:bool, rank: int) -> bool:
    if not dist.is_initialized():
        return stop_training
    t = torch.tensor([stop_training], dtype=torch.float32, device=f"cuda:{rank}")
    dist.broadcast(t, src=0)
    return bool(t.item())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_dataloader_standard(experiment_config: ExperimentConfig, rank=0, world_size=1):
    # token单独在另一个进程处理，解耦合
    logger.info(f"Rank {rank}: 创建MoonDataset")
    # 创建 MoonDataset
    train_dataset = MoonDataset(
        dataset_name=experiment_config.dataset_name,
        max_length=experiment_config.max_length
    )
    
    # DistributedSampler 工作原理：将数据均匀分配给多个GPU
    sampler = DistributedSampler(train_dataset,
                                num_replicas=world_size, # 总进程数
                                rank=rank,  # 当前进程排名
                                shuffle=True) if world_size > 1 else None # world_size大于1才使用sampler
    
    # DataLoader 配置优化
    num_workers = min(2, mp.cpu_count() // world_size)
    per_gpu_batch_size = experiment_config.batch_size // world_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,  # 禁用持久化worker，避免状态累积
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
        timeout=30  # 添加30秒超时
    )
    
    logger.info(f"Rank {rank}: DataLoader配置完成 - {num_workers} workers, batch_size {per_gpu_batch_size}")
        
    return train_loader, sampler

def get_dataloader_streaming(experiment_config: ExperimentConfig, stream_config: StreamConfig, rank=0, world_size=1):
    """创建模型和流式数据加载器"""
    
    logger.info(f"Rank {rank}: 创建流式数据加载系统")
    
    # 创建共享缓冲区管理器（所有进程共享）
    buffer_manager = SharedBufferManager(stream_config) # 使用主进程创建共享内存阶段的配置不能变
    
    # 创建流式数据集
    stream_dataloader = StreamingMoonDataset(
        buffer_manager=buffer_manager,
        rank=rank,
        world_size=world_size
    )
    
    logger.info(f"Rank {rank}: 流式数据加载系统配置完成")
        
    return stream_dataloader

class BaseExperimentResources(ABC):
    """基础实验资源容器（抽象基类）"""
    def __init__(
        self,
        model,
        optimizer,
        device,
        lr_scheduler,
        rank=0,
        world_size=1
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.world_size = world_size
    
    @abstractmethod
    def cleanup(self):
        """清理资源（抽象方法）"""
        pass

class StreamingExperimentResources(BaseExperimentResources):
    """流式训练资源容器"""
    def __init__(
        self,
        model,
        optimizer,
        device,
        lr_scheduler,
        streaming_dataloader,
        rank=0,
        world_size=1
    ):
        super().__init__(model, optimizer, device, lr_scheduler, rank, world_size)
        self.streaming_dataloader = streaming_dataloader
    
    def cleanup(self):
        """流式训练特有的清理逻辑"""
        self.streaming_dataloader.buffer_manager.close()

class StandardExperimentResources(BaseExperimentResources):
    """标准训练资源容器"""
    def __init__(
        self,
        model,
        optimizer,
        device,
        lr_scheduler,
        train_loader,
        sampler=None,
        rank=0,
        world_size=1
    ):
        super().__init__(model, optimizer, device, lr_scheduler, rank, world_size)
        self.train_loader = train_loader
        self.sampler = sampler

    def cleanup(self):
        """标准训练特有的清理逻辑"""
        # 标准训练没有特殊清理需求
        pass

def _create_resorce_standard(experiment_config, device, lr_scheduler, optimizer, model, rank=0, world_size=1):
    # 初始化所有资源
    train_loader, sampler = get_dataloader_standard(
        experiment_config = experiment_config,
        rank=rank,
        world_size=world_size
    )

    resorces_standard = StandardExperimentResources(model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
        sampler=sampler,
        rank=rank,
        world_size=world_size)
    return resorces_standard

def _create_resorce_streaming(experiment_config, stream_config, device, lr_scheduler, optimizer, model, rank=0, world_size=1):
    # 初始化所有资源
    streaming_dataloader = get_dataloader_streaming(
        experiment_config = experiment_config,
        stream_config = stream_config,
        rank=rank,
        world_size=world_size
    )

    resorces_standard = StreamingExperimentResources(model=model,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
        rank=rank,
        world_size=world_size,
        streaming_dataloader=streaming_dataloader)
    return resorces_standard

def compute_training_steps(experiment_config: ExperimentConfig):
    max_token_num=experiment_config.max_tokens
    tokens_per_step=experiment_config.batch_size*experiment_config.max_length
    return max_token_num//tokens_per_step

@contextmanager
def experiment_manager(experiment_config: ExperimentConfig, stream_config: StreamConfig = None, rank=0, world_size=1, training_mode="streaming"):
    """管理GPU、数据集、模型资源,管理日志打印"""
    step_func_name = experiment_config.step_func_name
    optimizer_name = experiment_config.optimizer_name
    loss_threshold = experiment_config.loss_threshold
    log_file_path = experiment_config.log_file_path
    lr = experiment_config.lr
    model_name = experiment_config.model_name
    dataset_name = experiment_config.dataset_name
    max_epochs = experiment_config.max_epochs
    max_position_embeddings = experiment_config.max_position_embeddings
    max_length = experiment_config.max_length
    batch_size = experiment_config.batch_size
  
    # 控制日志范围
    timestamp = get_timestamp()
    logger.remove()
    if rank == 0:
        sink_id = logger.add(
            f"{log_file_path}/{timestamp}_train_{step_func_name}_{dataset_name}_{model_name}_{optimizer_name}_lr{lr}_batch_size={batch_size}.log", 
            mode="w", level="INFO"
        )
    else:
        sink_id = logger.add(
            f"{log_file_path}/{timestamp}_rank{rank}_{step_func_name}.log", 
            mode="w", level="INFO"  # 改为INFO级别以便调试
        )
    
    # 打印工作进程初始化成功日志
    if world_size > 1:
        setup_ddp(rank, world_size)
        logger.info(f"🎯 Rank {rank}/{world_size} 初始化完成")
        logger.info(f"🖥️  当前GPU: {torch.cuda.current_device()}")
        logger.info(f"🌐 进程组: {dist.get_world_size()} 个进程")

    # 创建模型
    model = create_qwen_model(
        model_name=experiment_config.model_name,
        hidden_size=experiment_config.hidden_size,
        max_position_embeddings=experiment_config.max_position_embeddings
    )

    total_params = count_parameters(model)
    # 只在主进程显示启动信息
    if rank == 0:
        logger.info(f"🚀 开始实验: {optimizer_name}_{step_func_name}")
        logger.info(f"目标损失阈值: {loss_threshold}, 最大epoch数: {max_epochs}, 总参数量: {total_params}")
        logger.info(f"max_position_embeddings: {max_position_embeddings}, max_length: {max_length}")
        logger.info(f"使用DDP训练, 检测到 {world_size} 个GPU")
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # 获取优化器
    original_model = model.module if world_size > 1 else model
    optimizer = get_optimizer(
        step_func=STEP_MAP[step_func_name]["step_func"],
        optimizer_name=optimizer_name, 
        model=original_model,
        lr=lr, 
        wd=experiment_config.wd,
        block_num=experiment_config.block_num,
    )
    
    num_training_steps = compute_training_steps(experiment_config)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_training_steps//20,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
    
    if (training_mode == "streaming"):
        resources = _create_resorce_streaming(experiment_config, stream_config, device, lr_scheduler, optimizer, model, rank, world_size)
    elif (training_mode == "standard"):
        resources = _create_resorce_standard(experiment_config, device, lr_scheduler, optimizer, model, rank, world_size)
    else:
        assert 0, f"Training_mode {training_mode} not supported!"
    
    try:
        if rank == 0:
            logger.info("✅ 资源初始化完成")
        yield resources
        
    finally:
        # 清理资源
        if rank == 0:
            logger.info("🧹 清理实验资源...")
        if world_size > 1:
            cleanup_ddp()
        
        # 清理资源引用
        resources.cleanup()
        del resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if rank == 0:
            logger.info("✅ 资源清理完成")
        logger.remove(sink_id)

def train_worker(rank, world_size, experiment_config:ExperimentConfig, stream_config: StreamConfig = None, training_mode="streaming"):
    """DDP训练工作进程"""
    # experiment_manager创建实验所需全部资源，放到resources里面
    with experiment_manager(experiment_config, stream_config, rank, world_size, training_mode) as resources:
        # 初始化训练状态
        training_state = TrainingState(experiment_config)
        
        # 开始训练
        if training_mode is "standard":
            _run_standard_training_loop(resources, training_state, experiment_config, rank, world_size)
        elif training_mode is "streaming":
            _run_streaming_training_loop(resources, training_state, experiment_config, rank, world_size)
        # 训练结束处理
        _finalize_training(resources, training_state, experiment_config, rank)
        
        return training_state.cur_loss, training_state.current_interval_loss_sum

class TrainingState:
    """训练状态容器"""
    def __init__(self, experiment_config):
        # 训练统计
        self.cur_loss=0
        self.current_interval_loss_sum=0
        self.epoch_losses=0
        self.total_tokens_trained = 0
        self.step = 0
        self.epoch = 0
        self.skipped_batches = 0
        self.stop_training = False
        
        # 计算常量
        self.tokens_per_batch = experiment_config.batch_size * experiment_config.max_length
        
        # 吞吐量统计
        self.train_start_time = time.time()
        self.last_step_time = self.train_start_time
        self.step_count_since_last = 0
        self.tokens_since_last = 0

def _run_standard_training_loop(resources:StandardExperimentResources, training_state:TrainingState, experiment_config:ExperimentConfig, rank):
    """运行标准训练循环"""
    # 创建进度条（只在rank 0）
    if rank == 0:
        epoch_pbar = tqdm(
            total=experiment_config.max_tokens,
            desc=f"Epoch {training_state.epoch+1}",
            unit="tokens",
            ncols=120,
            position=0,
            leave=True,
        )
    
    while training_state.epoch < experiment_config.max_epochs and not training_state.stop_training:
        logger.info(f"Rank {rank}: 开始Epoch {training_state.epoch}")
        
        resources.sampler.set_epoch(training_state.epoch)
        total_batches = len(resources.train_loader)
        
        for batch in enumerate(resources.train_loader):
            # 检查跳过批次阈值
            if training_state.skipped_batches > total_batches * 0.05:
                logger.error(f"Rank {rank}: ❌ 跳过批次过多 ({training_state.skipped_batches}/{total_batches},"
                             " {training_state.skipped_batches/total_batches*100:.1f}%)，停止训练")
                raise
            
            # 检查token限制
            if (rank == 0):
                if training_state.total_tokens_trained >= experiment_config.max_tokens:
                    logger.info(f"Rank {rank}: 🎯 已达到目标token数 {experiment_config.max_tokens}, 停止训练")
                    training_state.stop_training = True
                    break
                
            success = _process_batch(resources, batch, training_state, experiment_config, rank)
            if not success:
                training_state.skipped_batches += 1
                continue
        
        # epoch结束处理
        _finalize_epoch(training_state, rank, total_batches, experiment_config.loss_threshold)
        
        # 同步停止信号
        training_state.stop_training = broadcast_stop_signal(training_state.stop_training, rank)
        training_state.epoch += 1
    
    if rank == 0:
        epoch_pbar.close()

STREAM_EPOCH_LEN=100
def _run_streaming_training_loop(resources, training_state:TrainingState, experiment_config:ExperimentConfig, rank):
    """运行流式训练循环"""
    # 创建进度条（只在rank 0）
    if rank == 0:
        epoch_pbar = tqdm(
            total=experiment_config.max_tokens,
            desc=f"Epoch {training_state.epoch+1}",
            unit="tokens",
            ncols=120,
            position=0,
            leave=True,
        )
    while not training_state.stop_training:
        training_state.epoch_losses=0
        while training_state.step < STREAM_EPOCH_LEN:
            # 获取下一个batch
            batch = resources.streaming_dataloader.get_next_batch()
            if batch is None:
                logger.info(f"Rank {rank}: 没有更多数据，停止训练")
                raise
                
            _process_batch(resources, batch, training_state, experiment_config, rank)
        if rank == 0 and training_state.total_tokens_trained >= experiment_config.max_tokens:
            training_state.stop_training=True
            training_state.epoch_losses /= STREAM_EPOCH_LEN
            # 同步停止信号
            training_state.stop_training = broadcast_stop_signal(training_state.stop_training, rank)
    
    if rank == 0:
        epoch_pbar.close()

def _process_batch(resources:BaseExperimentResources, batch, state:TrainingState, experiment_config:ExperimentConfig, rank):
    """处理单个batch的训练"""
    # 优化器清零
    resources.optimizer.zero_grad()
    
    # 数据转移到GPU
    batch = batch.to(resources.device, non_blocking=True)
    # 前向传播
    outputs = resources.model(input_ids=batch, labels=batch)
    loss = outputs.loss
    
    # 反向传播
    loss.backward()
    resources.optimizer.step()
    resources.lr_scheduler.step()
    
    current_loss = loss.item()
    state.current_interval_loss_sum += current_loss
    state.epoch_losses += current_loss
    
    # 更新统计
    state.total_tokens_trained += state.tokens_per_batch
    state.step += 1
    
    # 吞吐量统计
    if rank == 0:
        state.step_count_since_last += 1
        state.tokens_since_last += state.tokens_per_batch
        log_interval = 100
        # 日志记录
        if state.step % log_interval == 0:
            avg_loss = state.current_interval_loss_sum/log_interval
            state.current_interval_loss_sum=0
            _log_training_progress(state, experiment_config, rank, resources.optimizer.param_groups[0]['lr'], avg_loss)
    
    return True

def _log_training_progress(state:TrainingState, experiment_config:ExperimentConfig, rank, cur_lr, avg_loss):
    """记录训练进度"""
    progress_pct = state.total_tokens_trained/experiment_config.max_tokens*100 if experiment_config.max_tokens != float('inf') else 0
    skip_pct = state.skipped_batches*state.tokens_per_batch / experiment_config.max_tokens * 100 if state.skipped_batches > 0 else 0
    
    # 吞吐量计算
    throughput_info = ""
    if rank == 0:
        current_time = time.time()
        time_elapsed = current_time - state.last_step_time
        if time_elapsed > 0:
            tokens_per_sec = state.tokens_since_last / time_elapsed
            throughput_info = f" Throughput: {tokens_per_sec:.0f} tokens/sec"
            
            # 重置统计
            state.last_step_time = current_time
            state.step_count_since_last = 0
            state.tokens_since_last = 0

    logger.info(
        f"Rank {rank}: StepFunc: {experiment_config.step_func_name} "
        f"Epoch: {state.epoch} Step: {state.step}, "
        f"Tokens: {state.total_tokens_trained}/{experiment_config.max_tokens} ({progress_pct:.1f}%), {throughput_info}"
        f"Loss: {avg_loss:.4f} Skipped: {state.skipped_batches} ({skip_pct:.1f}%),"
        f"lr: {cur_lr:.5e}"
    )

def _finalize_epoch(state:TrainingState, rank, batches_per_epoch, loss_threshold):
    avg_epoch_loss = state.epoch_losses / batches_per_epoch
    final_skip_pct = state.skipped_batches / batches_per_epoch * 100
    logger.info(f"Rank {rank}: 📊 Epoch {state.epoch} 平均损失: {avg_epoch_loss:.4f},"
                f" 跳过批次: {state.skipped_batches}/{batches_per_epoch} ({final_skip_pct:.1f}%)")
    
    # 检查损失阈值
    if avg_epoch_loss < loss_threshold:
        state.stop_training = True
        logger.info(f"Rank {rank}: 🎯 已达到目标损失 {avg_epoch_loss:.4f}, 停止训练")

def _finalize_training(state:TrainingState, rank):
    # 最终统计
    final_throughput_info = ""
    if rank == 0:
        train_end_time = time.time()
        total_train_time = train_end_time - state.train_start_time
        if total_train_time > 0:
            overall_throughput = state.total_tokens_trained / total_train_time
            final_throughput_info = f" 总吞吐量: {overall_throughput:.0f} tokens/sec (总时间: {total_train_time:.1f}秒)"
    
    logger.info(f"Rank {rank}: 训练结束。总token数: {state.total_tokens_trained:,}, "
        f"stop_training={state.stop_training}, epoch_losses: {state.epoch_losses:.4f}{final_throughput_info}")

def run_experiment(experiment_config: ExperimentConfig, stream_config: StreamConfig = None, training_mode: str = "standard"):
    """运行单个实验，支持DDP和流式训练
    
    Args:
        experiment_config: 实验配置
        stream_config: 流式训练配置（仅流式训练需要）
        training_mode: 训练模式，"standard" 或 "streaming"
    """
    world_size = torch.cuda.device_count()
    
    # 验证参数
    if training_mode == "streaming" and stream_config is None:
        raise ValueError("流式训练需要提供 stream_config")

    # 共享内存创建器（只在流式训练中使用）
    shared_memory_creator = None
    loader_processes = []
    
    try:
        # 流式训练：在主进程创建共享内存
        if training_mode == "streaming" and stream_config is not None:
            shared_memory_creator = SharedMemoryCreator(stream_config)
            shared_info = shared_memory_creator.create_shared_memory() # 这里面已经包含了创建共享内存，初始化管理区域，并且关闭了fd，unmap
            
            # 启动数据加载进程（只在主进程）
            loader_processes = start_data_loaders(
                dataset_name=experiment_config.dataset_name,
                stream_config=stream_config,
                shared_info=shared_info,
                file_list=load_dataset_from_files(experiment_config.dataset_name),  # 需要获取文件列表
                tokenizer_name=experiment_config.model_name  # 使用config中的模型名称
            )
        
        # 运行训练
        if world_size > 1:
            # 多GPU使用DDP
            mp.spawn(
                train_worker,
                args=(world_size, experiment_config, training_mode, stream_config),
                nprocs=world_size,
                join=True
            )
            final_loss, losses = None, None
        else:
            # 单GPU直接调用训练函数
            final_loss, losses = train_worker(0, 1, experiment_config, training_mode, stream_config)
        
        return final_loss, losses
        
    finally:
        # 清理资源
        _cleanup_experiment_resources(loader_processes, shared_memory_creator, training_mode)

def _cleanup_experiment_resources(loader_processes, shared_memory_creator, training_mode):
    if training_mode == "streaming":
        # 使用专门的停止函数
        stop_data_loaders(loader_processes)
        # 清理共享内存
        if shared_memory_creator:
            logger.info("清理共享内存...")
            shared_memory_creator.cleanup()