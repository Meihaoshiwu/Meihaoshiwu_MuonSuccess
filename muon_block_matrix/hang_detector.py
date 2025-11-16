import os
import time
import threading
from loguru import logger
import torch.cuda
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