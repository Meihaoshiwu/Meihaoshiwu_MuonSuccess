import torch
import gc
import math
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
from .data import MoonDataset, load_dataset
from .optimizer import get_optimizer, STEP_MAP

# -------------- 工具函数 --------------
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_ddp(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def broadcast_stop_signal(stop_training: bool, rank: int) -> bool:
    if not dist.is_initialized():
        return stop_training
    t = torch.tensor([stop_training], dtype=torch.float32, device=f"cuda:{rank}")
    dist.broadcast(t, src=0)
    return bool(t.item())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_model_and_dataloader(model_name, dataset_name, hidden_size, max_position_embeddings=2048,
                            max_length=512, per_gpu_batch_size=32, rank=0, world_size=1):
    # token单独在另一个进程处理，解耦合
    # 创建 MoonDataset
    train_dataset = MoonDataset(
        dataset_name=dataset_name,
        max_length=max_length
    )
    
    # DistributedSampler 工作原理：将数据均匀分配给多个GPU
    # 在数据分配给各个进程之前进行全局打乱和划分。数据集只有一份，每个进程只是对索引进行打乱，没有数据集副本
    # 每个epoch进程间通信，set_epoch 设置种子，保证各进程之间打乱出来的索引一致
    # 根据rank每个进程取一部分索引，不重叠，拿着这些索引访问数据集
    sampler = DistributedSampler(train_dataset,
                                num_replicas=world_size, # 总进程数
                                rank=rank,  # 当前进程排名
                                shuffle=True) if world_size > 1 else None # world_size大于1才使用sampler
    
    # DataLoader 的工作流程：
    # 1. 从磁盘读取原始数据（文本、图像等）
    # 2. 数据预处理（分词、归一化、数据增强）
    # 3. 组成 batch
    # 4. 传输到 GPU
    # 前3步都在 CPU 上执行，只有第4步涉及 GPU
    num_workers = min(2, mp.cpu_count() // world_size)  # 每个GPU分到少量worker，本来已经做了数据并行的多进程
    train_loader = DataLoader( # 只涉及给分配好的数据打包，不涉及分配数据到GPU
        train_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,           # 使用多进程加载数据
        pin_memory=True,                   # 将数据直接加载到 GPU 可快速访问的"锁页内存"
        persistent_workers=(num_workers > 0),  # 保持worker进程，避免重复创建
        prefetch_factor=2 if num_workers > 0 else None,  # 每个 worker 预先加载的 batch 数量
        drop_last=True                     # 丢弃不完整的batch, 分布式训练中特别重要，确保所有 GPU 处理相同大小的 batch
    )
    
    logger.info(f"DataLoader configured with {num_workers} workers, per_gpu_batch_size {per_gpu_batch_size}")
    logger.info(f"Training with {world_size} GPU(s), rank {rank}")

    # 创建模型
    model = create_qwen_model(
        model_name=model_name,
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings
    )
        
    return model, train_loader, sampler

class ExperimentResources:
    """实验资源容器"""
    def __init__(
            self,
            model,
            train_loader,
            optimizer,
            device,
            lr_scheduler,
            sampler=None,
            rank=0,
            world_size=1):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.sampler = sampler
        self.rank = rank
        self.world_size = world_size

@contextmanager
def experiment_manager(experiment_config: ExperimentConfig, rank=0, world_size=1):
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
            f"{log_file_path}/{timestamp}_train_{step_func_name}_{dataset_name}_{model_name}_{optimizer_name}_lr{lr}.log", 
            mode="w", level="INFO"
        )
    else:
        sink_id = logger.add(
            f"{log_file_path}/{timestamp}_rank{rank}_{step_func_name}.log", 
            mode="w", level="ERROR"
        )
    
    # 打印工作进程初始化成功日志
    if world_size > 1:
        setup_ddp(rank, world_size)
        logger.info(f"🎯 Rank {rank}/{world_size} 初始化完成")
        logger.info(f"🖥️  当前GPU: {torch.cuda.current_device()}")
        logger.info(f"🌐 进程组: {dist.get_world_size()} 个进程")

    # 初始化所有资源
    model, train_loader, sampler = get_model_and_dataloader(
        model_name=model_name,
        dataset_name=dataset_name,
        hidden_size=experiment_config.hidden_size,
        max_position_embeddings=max_position_embeddings,
        max_length=max_length,
        per_gpu_batch_size=batch_size//world_size, # 多进程将batch分给多个GPU
        rank=rank,
        world_size=world_size
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
        wd=experiment_config.wd
    )
    
    tokens_per_step = batch_size * max_length  # batch_size是全局batch大小
    
    # 计算总步数（向上取整）
    max_tokens = experiment_config.max_tokens
    num_training_steps = int(math.ceil(max_tokens / tokens_per_step))
    
    estimated_epochs = int(math.ceil(num_training_steps / len(train_loader)))
    if rank == 0:
        logger.info(f"📊 基于token限制计算: {max_tokens:,} tokens")
        logger.info(f"📊 每个step处理: {tokens_per_step:,} tokens")
        logger.info(f"📊 总训练步数: {num_training_steps:,} steps")
        logger.info(f"📊 估计epoch数: ~{estimated_epochs} epochs")
    
    # 计算warmup步数
    num_warmup_steps = num_training_steps // 20
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
    
    # 封装资源
    resources = ExperimentResources(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
        sampler=sampler,
        rank=rank,
        world_size=world_size
    )
    
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
        del resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if rank == 0:
            logger.info("✅ 资源清理完成")
        logger.remove(sink_id)

def train_worker(rank, world_size, experiment_config):
    """DDP训练工作进程"""
    with experiment_manager(experiment_config, rank, world_size) as resources:
        model = resources.model
        train_loader = resources.train_loader
        optimizer = resources.optimizer
        device = resources.device
        lr_scheduler = resources.lr_scheduler
        sampler = resources.sampler
        step_func_name = experiment_config.step_func_name
        
        model.train()
        losses = []
        total_tokens_trained = 0
        batch_size = experiment_config.batch_size # 这是全量的batch大小
        max_length = experiment_config.max_length
        max_tokens = experiment_config.max_tokens
        
        # 计算每个batch的token数
        tokens_per_batch = batch_size * max_length
        
        epoch = 0
        stop_training = False
        
        # 外层循环改为基于epoch，内层检查token数
        while epoch < experiment_config.max_epochs and not stop_training:
            if sampler:
                sampler.set_epoch(epoch)
                
            epoch_losses = []
            
            if rank == 0:
                epoch_pbar = tqdm(
                    total=max_tokens,
                    desc=f"Epoch {epoch+1}",
                    unit="tokens",
                    ncols=100,
                    position=0,
                    leave=True,
                )
            
            for step, batch in enumerate(train_loader):
                # 检查是否达到token限制
                if total_tokens_trained >= max_tokens:
                    logger.info(f"🎯 已达到目标token数 {max_tokens}, 停止训练")
                    stop_training = True
                    break
                
                optimizer.zero_grad()
                batch = batch.to(device)
                input_ids = batch
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                current_loss = loss.item()
                epoch_losses.append(current_loss)
                
                # 更新token计数
                total_tokens_trained += tokens_per_batch
                
                if rank == 0:
                    epoch_pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'progress': f'{total_tokens_trained/max_tokens*100:.1f}%' if max_tokens != float('inf') else 'N/A',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.5e}'
                    })
                    epoch_pbar.update(tokens_per_batch)
                    
                if step % 1000 == 0:
                    progress_pct = total_tokens_trained/max_tokens*100 if max_tokens != float('inf') else 0
                    logger.info(
                        f"StepFunc: {step_func_name} Epoch: {epoch} Step: {step} Rank: {rank}"
                        f"Tokens: {total_tokens_trained}/{max_tokens} ({progress_pct:.1f}%) "
                        f"Loss: {current_loss:.4f}, lr = {optimizer.param_groups[0]["lr"]}"
                    )
            
            if rank == 0:
                epoch_pbar.close()
                
                # 计算epoch平均损失
                if epoch_losses:  # 避免除零
                    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                    losses.append(avg_epoch_loss)
                    logger.info(f"📊 {step_func_name} - Epoch {epoch} 平均损失: {avg_epoch_loss:.4f}")
                    
                    # 保留loss阈值检查作为备选停止条件
                    if avg_epoch_loss < experiment_config.loss_threshold:
                        stop_training = True
                        logger.info(f"🎯 已达到目标损失 {avg_epoch_loss:.4f}, 停止训练")

            # 检查主进程已经达到停止条件,src=0指定了使用主进程的stop向量
            stop_training = broadcast_stop_signal(stop_training, rank)
            
            epoch += 1
        
        final_loss = losses[-1] if losses else float('inf')
        logger.info(f"训练结束。总token数: {total_tokens_trained:,}, stop_training={stop_training},rank={rank}"
                    " final_loss: {final_loss:.4f}, avg_epoch_loss = {avg_epoch_loss}")
        return final_loss, losses

def run_experiment(experiment_config: ExperimentConfig):
    """运行单个实验，支持DDP"""
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # 多GPU使用DDP
        mp.spawn(
            train_worker,
            args=(world_size, experiment_config),
            nprocs=world_size,
            join=True
        )
        return None, None
    else:
        # 单GPU直接调用训练函数
        return train_worker(0, 1, experiment_config)