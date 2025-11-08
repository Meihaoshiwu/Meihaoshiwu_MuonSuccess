import gc
import time
import threading

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
from .data import MoonDataset, load_dataset
from .optimizer import get_optimizer, STEP_MAP

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
    logger.info(f"Rank {rank}: 创建MoonDataset")
    # 创建 MoonDataset
    train_dataset = MoonDataset(
        dataset_name=dataset_name,
        max_length=max_length
    )
    
    # DistributedSampler 工作原理：将数据均匀分配给多个GPU
    sampler = DistributedSampler(train_dataset,
                                num_replicas=world_size, # 总进程数
                                rank=rank,  # 当前进程排名
                                shuffle=True) if world_size > 1 else None # world_size大于1才使用sampler
    
    # DataLoader 配置优化
    num_workers = min(2, mp.cpu_count() // world_size)
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
            mode="w", level="INFO"  # 改为INFO级别以便调试
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
        per_gpu_batch_size=batch_size//world_size,
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
    
    num_training_steps = len(train_loader) * max_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(100, num_training_steps // 10),
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

class HangDetector:
    """hang检测器"""
    def __init__(self, rank, timeout=300):  # 5分钟超时
        self.rank = rank
        self.timeout = timeout
        self.last_activity_time = time.time()
        self.active_step = None
        self.monitor_thread = None
        self.enabled = True
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitor_thread is None:
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info(f"Rank {self.rank}: Hang检测器已启动，超时时间 {self.timeout}秒")
    
    def update_activity(self, step_name):
        """更新活动状态"""
        self.last_activity_time = time.time()
        self.active_step = step_name
        logger.debug(f"Rank {self.rank}: 活动更新 - {step_name}")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.enabled:
            current_time = time.time()
            idle_time = current_time - self.last_activity_time
            
            if idle_time > self.timeout:
                logger.error(f"Rank {self.rank}: ❌ 检测到hang！当前步骤: {self.active_step}, "
                           f"已空闲 {idle_time:.0f}秒 (超时: {self.timeout}秒)")
                
                # 输出详细诊断信息
                self._dump_diagnostic_info()
                
                # 强制退出进程
                os._exit(1)
            
            # 每分钟记录一次状态
            if int(current_time) % 60 == 0:
                logger.info(f"Rank {self.rank}: 监控状态 - 当前步骤: {self.active_step}, "
                          f"空闲时间: {idle_time:.0f}秒")
            
            time.sleep(10)  # 每10秒检查一次
    
    def _dump_diagnostic_info(self):
        """输出诊断信息"""
        import traceback
        import sys
        
        logger.error(f"Rank {self.rank}: === HANG诊断信息 ===")
        logger.error(f"当前步骤: {self.active_step}")
        logger.error(f"进程PID: {os.getpid()}")
        logger.error(f"父进程PID: {os.getppid()}")
        logger.error(f"活动线程数: {threading.active_count()}")
        
        # 输出所有线程的堆栈
        for thread_id, stack in sys._current_frames().items():
            logger.error(f"线程 {thread_id} 堆栈:")
            for filename, lineno, name, line in traceback.extract_stack(stack):
                logger.error(f"  {filename}:{lineno} in {name}")
        
        # 输出GPU内存信息
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                logger.error(f"GPU内存使用: {gpu_memory:.2f} GB")
            except:
                logger.error("无法获取GPU内存信息")
        
        logger.error(f"Rank {self.rank}: === 诊断信息结束 ===")

# 在训练进程中添加hang检测
def train_worker(rank, world_size, experiment_config):
    """DDP训练工作进程"""
    # 创建hang检测器
    hang_detector = HangDetector(rank, timeout=300)  # 5分钟超时
    hang_detector.start_monitoring()
    
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
        batch_size = experiment_config.batch_size
        max_length = experiment_config.max_length
        max_tokens = experiment_config.max_tokens
        
        # 计算每个batch的token数
        tokens_per_batch = batch_size * max_length
        tokens_per_epoch = len(train_loader) * tokens_per_batch
        epoch = 0
        stop_training = False
        
        # 外层循环改为基于epoch，内层检查token数
        while epoch < experiment_config.max_epochs and not stop_training:
            logger.info(f"Rank {rank}: 开始Epoch {epoch}")
            hang_detector.update_activity(f"Epoch {epoch} 开始")
            
            if sampler:
                sampler.set_epoch(epoch)
                
            epoch_losses = []
            
            # 添加跳过批次计数器
            skipped_batches = 0
            total_batches = len(train_loader)
            skip_threshold = total_batches * 0.05  # 5%阈值
            
            if rank == 0:
                epoch_pbar = tqdm(
                    total=tokens_per_epoch,
                    desc=f"Epoch {epoch+1}",
                    unit="tokens",
                    ncols=100,
                    position=0,
                    leave=True,
                )
            
            for step, batch in enumerate(train_loader):
                hang_detector.update_activity(f"Epoch {epoch} Step {step} - 获取batch")
                
                # 检查跳过批次是否超过阈值
                if skipped_batches > skip_threshold:
                    logger.error(f"Rank {rank}: ❌ 跳过批次过多 ({skipped_batches}/{total_batches}, {skipped_batches/total_batches*100:.1f}%)，停止训练")
                    stop_training = True
                    break
                    
                # 检查是否达到token限制
                if total_tokens_trained >= max_tokens:
                    logger.info(f"Rank {rank}: 🎯 已达到目标token数 {max_tokens}, 停止训练")
                    stop_training = True
                    break
                
                try:
                    hang_detector.update_activity(f"Epoch {epoch} Step {step} - 优化器清零")
                    optimizer.zero_grad()
                    
                    hang_detector.update_activity(f"Epoch {epoch} Step {step} - 数据转移到GPU")
                    batch = batch.to(device)
                    input_ids = batch
                    
                    hang_detector.update_activity(f"Epoch {epoch} Step {step} - 前向传播")
                    logger.debug(f"Rank {rank}: Epoch {epoch} Step {step} 开始前向传播")
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss
                    
                    hang_detector.update_activity(f"Epoch {epoch} Step {step} - 反向传播")
                    loss.backward()
                    
                    hang_detector.update_activity(f"Epoch {epoch} Step {step} - 优化器步骤")
                    optimizer.step()
                    
                    hang_detector.update_activity(f"Epoch {epoch} Step {step} - 学习率调度")
                    lr_scheduler.step()
                    
                    current_loss = loss.item()
                    epoch_losses.append(current_loss)
                    
                    # 更新token计数
                    total_tokens_trained += tokens_per_batch
                    
                    if rank == 0:
                        epoch_pbar.set_postfix({
                            'loss': f'{current_loss:.4f}',
                            'progress': f'{total_tokens_trained/max_tokens*100:.1f}%' if max_tokens != float('inf') else 'N/A',
                            'lr': f'{optimizer.param_groups[0]["lr"]:.5e}',
                            'skipped': f'{skipped_batches}'  # 显示跳过的批次数
                        })
                        epoch_pbar.update(tokens_per_batch)
                        
                    if step % 100 == 0:
                        progress_pct = total_tokens_trained/max_tokens*100 if max_tokens != float('inf') else 0
                        skip_pct = skipped_batches / (step + 1) * 100 if step > 0 else 0
                        logger.info(
                            f"Rank {rank}: StepFunc: {step_func_name} Epoch: {epoch} Step: {step} "
                            f"Tokens: {total_tokens_trained}/{max_tokens} ({progress_pct:.1f}%) "
                            f"Loss: {current_loss:.4f} Skipped: {skipped_batches} ({skip_pct:.1f}%)"
                        )
                        
                except RuntimeError as e:
                    if "DataLoader" in str(e) or "timeout" in str(e).lower():
                        skipped_batches += 1
                        skip_pct = skipped_batches / (step + 1) * 100
                        logger.warning(f"Rank {rank}: DataLoader超时，跳过step {step}, 已跳过 {skipped_batches} 批次 ({skip_pct:.1f}%)")
                        
                        # 检查是否接近阈值
                        if skipped_batches > skip_threshold * 0.8:  # 达到阈值的80%时警告
                            logger.warning(f"Rank {rank}: ⚠️ 跳过批次接近阈值 ({skipped_batches}/{skip_threshold:.0f})")
                            
                        continue  # 跳过当前batch，继续下一个
                    else:
                        raise  # 重新抛出其他异常
            
            if rank == 0:
                epoch_pbar.close()
                
                # 计算epoch平均损失
                if epoch_losses:
                    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                    losses.append(avg_epoch_loss)
                    final_skip_pct = skipped_batches / total_batches * 100 if total_batches > 0 else 0
                    logger.info(f"Rank {rank}: 📊 {step_func_name} - Epoch {epoch} 平均损失: {avg_epoch_loss:.4f}, 跳过批次: {skipped_batches}/{total_batches} ({final_skip_pct:.1f}%)")
                    
                    # 保留loss阈值检查作为备选停止条件
                    if avg_epoch_loss < experiment_config.loss_threshold:
                        stop_training = True
                        logger.info(f"Rank {rank}: 🎯 已达到目标损失 {avg_epoch_loss:.4f}, 停止训练")

            # 同步停止信号
            logger.debug(f"Rank {rank}: 同步停止信号, 当前stop_training={stop_training}")
            hang_detector.update_activity("同步停止信号")
            stop_training = broadcast_stop_signal(stop_training, rank)
            logger.info(f"Rank {rank}: Epoch {epoch} 完成, 累计token: {total_tokens_trained}")
            
            epoch += 1
        
        # 停止hang检测器
        hang_detector.enabled = False
        
        final_loss = losses[-1] if losses else float('inf')
        avg_epoch_loss = losses[-1] if losses else float('inf')
        logger.info(f"Rank {rank}: 训练结束。总token数: {total_tokens_trained:,}, stop_training={stop_training}, "
                   f"final_loss: {final_loss:.4f}, avg_epoch_loss = {avg_epoch_loss:.4f}")
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