import ctypes
import os
import numpy as np
import torch
from typing import Dict, Any

# 加载编译好的C扩展
try:
    lib = ctypes.CDLL('./libshm_mutex.so')
except Exception as e:
    print(f"Warning: Could not load libshm_mutex.so: {e}")
    lib = None

# 定义C结构体
class ShmHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_int),
        ("version", ctypes.c_int),
        ("num_buffers", ctypes.c_int),
        ("padding", ctypes.c_byte * 52)  # 填充到64字节
    ]

class BufferControl(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int),
        ("mutex", ctypes.c_byte * 40),  # pthread_mutex_t
        ("padding", ctypes.c_byte * 20)  # 填充到64字节
    ]

class StreamConfig:
    """流式数据配置"""
    def __init__(
        self,
        batches_per_chunk: int = 1024,
        num_buffers: int = 16,
        num_loaders: int = 4,
        max_length: int = 512,
        batch_size: int = 32,
        loader_rest_threshold: float = 0.8,
        world_size: int =8, # 缓冲区被多少个训练进程读取
    ):
        assert (batch_size % world_size) is 0
        self.batches_per_chunk = batches_per_chunk
        self.num_buffers = num_buffers
        self.num_loaders = num_loaders
        self.max_length = max_length
        self.batch_size = batch_size
        self.loader_rest_threshold = loader_rest_threshold
        self.world_size = world_size

TOKEN_ID_BYTES = 8  # 与训练进程保持一致，训练进程需要长度为8B，避免转换
class SharedMemoryCreator:
    """共享内存创建器 - 只在主进程使用"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.control_shm_name = f"/muon_control_{os.getpid()}"
        self.data_shm_name = f"/muon_data_{os.getpid()}"
        
        # 计算大小
        self.control_size = ctypes.sizeof(ShmHeader) + config.num_buffers * ctypes.sizeof(BufferControl)
        self.data_size = config.num_buffers * config.batches_per_chunk * config.batch_size * config.max_length * TOKEN_ID_BYTES
    
    def create_shared_memory(self) -> Dict[str, Any]:
        """创建共享内存资源"""
        if lib is None:
            raise RuntimeError("C library not loaded")
            
        result = lib.init_shared_memory(
            self.control_shm_name.encode(),
            self.data_shm_name.encode(),
            self.config.num_buffers,
            self.data_size
        )
        
        if result != 0:
            raise RuntimeError("Failed to create shared memory")
        
        return {
            'control_shm_name': self.control_shm_name,
            'data_shm_name': self.data_shm_name,
            'control_size': self.control_size,
            'data_size': self.data_size
        }
    
    def cleanup(self):
        """清理共享内存"""
        try:
            # 清理共享内存文件
            control_path = f"/dev/shm/{self.control_shm_name[1:]}"
            data_path = f"/dev/shm/{self.data_shm_name[1:]}"
            if os.path.exists(control_path):
                os.unlink(control_path)
            if os.path.exists(data_path):
                os.unlink(data_path)
        except Exception as e:
            print(f"Warning: Failed to cleanup shared memory: {e}")

class SharedBufferManager:
    """共享缓冲区管理器 - 所有进程使用"""
    
    def __init__(self, config: StreamConfig, shared_info: Dict[str, Any]):
        self.config = config
        self.shared_info = shared_info
        
        if lib is None:
            raise RuntimeError("C library not loaded")
        
        # Attach到共享内存
        self.control_addr = lib.attach_shared_memory(
            shared_info['control_shm_name'].encode(),
            shared_info['control_size']
        )
        self.data_addr = lib.attach_shared_memory(
            shared_info['data_shm_name'].encode(), 
            shared_info['data_size']
        )
        
        if not self.control_addr or not self.data_addr:
            raise RuntimeError("Failed to attach shared memory")
        
        # 获取控制块指针
        self.header = ShmHeader.from_address(self.control_addr)
        self.controls = (BufferControl * self.config.num_buffers).from_address(
            ctypes.addressof(self.header) + ctypes.sizeof(ShmHeader)
        )
    
    def get_available_write_buffer(self) -> int:
        """获取可写缓冲区ID"""
        for i in range(self.config.num_buffers):
            if self._try_lock_buffer(i):
                try:
                    if self.controls[i].state == 0:  # 空闲
                        self.controls[i].state = 1   # 写入中
                        return i
                    else:
                        self._unlock_buffer(i)
                except Exception:
                    self._unlock_buffer(i)
                    raise
        return None
    
    def _try_lock_buffer(self, buffer_id: int) -> bool:
        """尝试锁定缓冲区（非阻塞）"""
        result = lib.try_lock_mutex(ctypes.byref(self.controls[buffer_id].mutex))
        return result == 0  # 0表示成功
    
    def _unlock_buffer(self, buffer_id: int):
        """解锁缓冲区"""
        lib.unlock_mutex(ctypes.byref(self.controls[buffer_id].mutex))
    
    def release_write_buffer(self, buffer_id: int):
        """释放写入缓冲区，标记为满"""
        if self._try_lock_buffer(buffer_id):
            try:
                self.controls[buffer_id].state = 2  # 满
            finally:
                self._unlock_buffer(buffer_id)
    
    def get_available_read_buffer(self) -> int:
        """获取可读的缓冲区ID"""
        for i in range(self.config.num_buffers):
            if self._try_lock_buffer(i):
                try:
                    if self.controls[i].state == 2:  # 满
                        self.controls[i].state = 3   # 读取中
                        return i
                    else:
                        self._unlock_buffer(i)
                except Exception:
                    self._unlock_buffer(i)
                    raise
        return None
    
    def release_read_buffer(self, buffer_id: int):
        """释放读取缓冲区，标记为空"""
        if self._try_lock_buffer(buffer_id):
            try:
                self.controls[buffer_id].state = 0  # 空
            finally:
                self._unlock_buffer(buffer_id)
    
    def write_to_buffer(self, buffer_id: int, token_data: torch.Tensor):
        """写入数据到缓冲区"""
        chunk_array = token_data.numpy().flatten().astype(np.int32)
        
        # 计算数据偏移量
        buffer_size = self.config.batches_per_chunk * self.config.batch_size * self.config.max_length * TOKEN_ID_BYTES
        data_offset = buffer_id * buffer_size
        
        # 获取数据指针
        data_ptr = ctypes.cast(self.data_addr + data_offset, ctypes.POINTER(ctypes.c_int32))
        
        # 批量拷贝
        for i in range(len(chunk_array)):
            data_ptr[i] = chunk_array[i]
        
        self.release_write_buffer(buffer_id)
    
    def read_from_buffer(self, buffer_id: int, rank_id: int) -> torch.Tensor:
        """从缓冲区读取指定rank的数据部分"""
        # 计算整个chunk的大小和当前rank的数据偏移量
        total_samples = self.config.batches_per_chunk * self.config.batch_size * self.config.max_length
        samples_per_rank = total_samples // self.config.world_size
        
        # 计算整个chunk的偏移量和当前rank的偏移量
        buffer_offset = buffer_id * total_samples * TOKEN_ID_BYTES  # 整个chunk的偏移量
        rank_offset = buffer_offset + rank_id * samples_per_rank * TOKEN_ID_BYTES  # 当前rank的偏移量
        
        # 获取当前rank数据部分的指针
        data_ptr = ctypes.cast(self.data_addr + rank_offset, ctypes.POINTER(ctypes.c_int32))
        
        # 转换为numpy数组（只读取当前rank的数据）
        rank_array = np.ctypeslib.as_array(data_ptr, (samples_per_rank,))
        
        # 重塑为正确的形状 [batches_per_chunk, batch_size//world_size, max_length]
        batches_per_chunk = self.config.batches_per_chunk
        partial_batch_size = self.config.batch_size // self.config.world_size
        max_length = self.config.max_length
        
        try:
            reshaped_data = rank_array.reshape(batches_per_chunk, partial_batch_size, max_length)
            return torch.from_numpy(reshaped_data.copy())
        except ValueError as e:
            print(f"Buffer {buffer_id} rank {rank_id} reshape failed: {e}")
            return None
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区状态统计"""
        stats = {'empty': 0, 'writing': 0, 'full': 0, 'reading': 0}
        
        for i in range(self.config.num_buffers):
            state = self.controls[i].state
            if state == 0:
                stats['empty'] += 1
            elif state == 1:
                stats['writing'] += 1
            elif state == 2:
                stats['full'] += 1
            elif state == 3:
                stats['reading'] += 1
        
        stats['total_buffers'] = self.config.num_buffers
        stats['full_ratio'] = stats['full'] / stats['total_buffers']
        return stats
    
    def close(self):
        """关闭共享内存连接"""
        if hasattr(self, 'control_addr') and self.control_addr:
            lib.munmap(self.control_addr, self.shared_info['control_size'])
        if hasattr(self, 'data_addr') and self.data_addr:
            lib.munmap(self.data_addr, self.shared_info['data_size'])