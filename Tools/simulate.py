import math
import time
import torch
from loguru import logger
import os
from datetime import datetime

# 导入所有需要的函数（假设这些函数已经在其他地方定义）
# 这里只是示例，实际使用时需要导入这些函数
# from your_module import step_default, step_row_block, step_column_block, ...

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if X.size(0) > X.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
# matrix_block_num在这里没用，只是为了保持回调函数一致性
def step_default(G, steps, matrix_block_num):
    start_process_time = time.time()
    result = zeropower_via_newtonschulz5(G, steps)
    end_process_time = time.time()
    process_block_time = end_process_time - start_process_time
    return result, process_block_time, 0

def process_block(blocked_matrix, steps):
    if torch.norm(blocked_matrix) > 1e-7:
        orthogonalized_block = zeropower_via_newtonschulz5(blocked_matrix, steps)
        return orthogonalized_block
    else:
        return blocked_matrix

def step_column_block(G, steps, matrix_block_num: int):
    """
    将矩阵的列分成matrix_block_num块，每块独立进行正交化处理
    """
    result = torch.zeros_like(G) # 用于保存结果
    cols = G.shape[1]  # 总列数
    
    # 检查是否支持这么多分块
    if matrix_block_num > cols:
        logger.info(f"Warning: 矩阵只有{cols}列，但要求分成{matrix_block_num}块，将作为整体处理")
        return process_block(G, steps), 0, 0
    
    block_size = cols // matrix_block_num  # 每块的列数

    start_process_time = time.time()
    for i in range(matrix_block_num-1):
        start_col = i * block_size
        end_col = (i + 1) * block_size
        column_block = G[:, start_col:end_col] # 取所有行，[start_col,end_col)列
        result[:, start_col:end_col] = process_block(column_block, steps)
    
    # 处理最后一块（可能包含剩余的列）
    start_col = (matrix_block_num-1) * block_size
    last_block = G[:, start_col:]
    result[:, start_col:] = process_block(last_block, steps)
    
    end_process_time = time.time()
    process_block_time = end_process_time - start_process_time
    return result, process_block_time, 0

def step_row_block(G, steps, matrix_block_num: int):
    """
    将矩阵的行分成matrix_block_num块，每块独立进行正交化处理
    """
    result = torch.zeros_like(G)
    rows = G.shape[0]  # 总行数
    
    # 检查是否支持这么多分块
    if matrix_block_num > rows:
        logger.info(f"Warning: 矩阵只有{rows}行，但要求分成{matrix_block_num}块，将作为整体处理")
        return process_block(G, steps), 0, 0
    
    block_size = rows // matrix_block_num  # 每块的行数
    start_process_time = time.time()
    # 处理前row_num-1个完整的块
    for i in range(matrix_block_num-1):
        start_row = i * block_size
        end_row = (i + 1) * block_size
        row_block = G[start_row:end_row, :]
        result[start_row:end_row, :] = process_block(row_block, steps)
    
    # 处理最后一块（可能包含剩余的行）
    start_row = (matrix_block_num-1) * block_size
    last_block = G[start_row:, :]
    result[start_row:, :] = process_block(last_block, steps)
    
    end_process_time = time.time()
    process_block_time = end_process_time - start_process_time
    return result, process_block_time, 0

# 横纵都切分
def step_block_matrix_flexible(G, steps, matrix_block_num: int):
    """
    更灵活的分块函数，可以处理非平方数的分块数
    例如：block_num=12 -> 可能会分成3x4或4x3等
    """
    result = torch.zeros_like(G)
    rows, cols = G.shape
    
    start_process_time = time.time()
    # 寻找最接近平方根的两个因数
    row_blocks, col_blocks = -1, -1
    sqrt_block = int(math.sqrt(matrix_block_num))
    for i in range(sqrt_block, 1, -1): # 至少不退化到列分块
        if matrix_block_num % i == 0:
            row_blocks, col_blocks = i, matrix_block_num // i
            break
    
    if row_blocks < 0:
        logger.info(f"Warning: 无法将{matrix_block_num}分解为合适的因数，将作为整体处理")
        return process_block(G, steps), 0, 0

    if rows // row_blocks < 1 or cols // col_blocks < 1:
        logger.info(f"Warning: 分块后某些块没有元素，将作为整体处理")
        return process_block(G, steps), 0, 0
    
    row_block_size = rows // row_blocks
    col_block_size = cols // col_blocks
    
    for i in range(row_blocks):
        for j in range(col_blocks):
            # 计算当前块的行范围
            start_row = i * row_block_size
            end_row = (i + 1) * row_block_size if i < row_blocks - 1 else rows
            
            # 计算当前块的列范围
            start_col = j * col_block_size
            end_col = (j + 1) * col_block_size if j < col_blocks - 1 else cols
            
            # 处理当前块
            block = G[start_row:end_row, start_col:end_col]
            result[start_row:end_row, start_col:end_col] = process_block(block, steps)
    
    end_process_time = time.time()
    process_block_time = end_process_time - start_process_time
    return result, process_block_time, 0

def estimate_svd_weights_and_process(G, steps, matrix_block_num: int):
    """
    对原始矩阵G分块估计奇异值之和，得到权重
    然后将权重乘到处理之后的矩阵上
    """
    rows = G.shape[0]  # 总行数
    
    # 矩阵太小没必要分块
    if matrix_block_num*matrix_block_num > rows:
        logger.info(f"Warning: 矩阵{rows}行，要求分成{matrix_block_num}块，将作为整体处理")
        return process_block(G, steps), 0, 0

    block_size = rows // matrix_block_num
    num_samples=20
    start_estimate = time.time()
    original_blocks = []
    for i in range(matrix_block_num):
        if i < matrix_block_num-1:
            start_row = i * block_size
            end_row = (i + 1) * block_size
            block = G[start_row:end_row, :]
        else:
            start_row = (matrix_block_num-1) * block_size
            block = G[start_row:, :]
        original_blocks.append(block)
    
    # 2. 对原始矩阵的每个块估计奇异值之和
    sv_estimates = []
    sv_accumulate = []
    for block in original_blocks:
        sv_est = randomized_nuclear_norm_estimate_fast(block, num_samples)
        sv_estimates.append(sv_est)
    for block in original_blocks:
        _, max_sv = singular_value_range(block)
        sv_accumulate.append(max_sv)
    logger.info(f"sv_estimates (随机核范数估计值): {[f'{x:.3e}' for x in sv_estimates]}")
    logger.info(f"sv_accumulate (每个块的最大奇异值): {[f'{x:.3e}' for x in sv_accumulate]}")
    
    # 3. 计算权重（每个块的奇异值之和占总和的比例）
    sv_estimates_tensor = torch.tensor(sv_estimates, device=G.device, dtype=G.dtype)
    # 检查奇异值之和是否为零，避免除零错误
    if torch.sum(sv_estimates_tensor) != 0:
        weights = sv_estimates_tensor / torch.sum(sv_estimates_tensor)
    else:
        weights = torch.ones_like(sv_estimates_tensor) / len(sv_estimates_tensor)
        logger.info("警告：所有块的奇异值估计都为零，使用均匀权重")
    end_estimate = time.time()
    estimate_time = end_estimate - start_estimate
    
    # 4. 再次处理每个块，但这次使用原始矩阵计算好的权重
    start_process_block = time.time()
    result = torch.zeros_like(G)
    for i in range(matrix_block_num):
        if i < matrix_block_num-1:
            start_row = i * block_size
            end_row = (i + 1) * block_size
            # 对每个块进行正交化处理
            processed_block = process_block(original_blocks[i], steps)
            # 乘以原始矩阵计算得到的权重
            weighted_block = processed_block * weights[i]
            result[start_row:end_row, :] = weighted_block
        else:
            start_row = (matrix_block_num-1) * block_size
            processed_block = process_block(original_blocks[matrix_block_num-1], steps)
            weighted_block = processed_block * weights[matrix_block_num-1]
            result[start_row:, :] = weighted_block
    end_process_block = time.time()
    process_block_time = end_process_block - start_process_block
    
    return result, process_block_time, estimate_time

def randomized_nuclear_norm_estimate_fast(A, num_samples=10):
    """
    快速随机估计矩阵A的奇异值之和
    生成num_samples个随机向量，计算A*z的范数，求平均后乘以n
    """
    n = A.shape[1]
    device = A.device
    
    # 生成随机向量
    Z = torch.randn(n, num_samples, device=device)
    Z = Z / torch.norm(Z, dim=0, keepdim=True)
    
    # 计算A*Z
    AZ = A @ Z
    
    # 计算每列的范数，求平均后乘以n
    norms = torch.norm(AZ, dim=0)
    estimate = torch.mean(norms).item() * n
    
    return estimate

# 谱范数归一化，分块正交之后乘以1/matrix_block_num
def step_row_block_spectral_normalization(G, steps, matrix_block_num: int):
    """
    将矩阵的行分成matrix_block_num块，每块独立进行正交化处理
    """
    result = torch.zeros_like(G)
    rows = G.shape[0]  # 总行数
    
    # 检查是否支持这么多分块
    if matrix_block_num > rows:
        logger.info(f"Warning: 矩阵只有{rows}行，但要求分成{matrix_block_num}块，将作为整体处理")
        return process_block(G, steps), 0, 0
    
    block_size = rows // matrix_block_num  # 每块的行数
    start_process_block = time.time()
    # 处理前row_num-1个完整的块
    for i in range(matrix_block_num-1):
        start_row = i * block_size
        end_row = (i + 1) * block_size
        row_block = G[start_row:end_row, :]
        result[start_row:end_row, :] = process_block(row_block, steps)
    
    # 处理最后一块（可能包含剩余的行）
    start_row = (matrix_block_num-1) * block_size
    last_block = G[start_row:, :]
    result[start_row:, :] = process_block(last_block, steps)
    
    end_process_block = time.time()
    process_block_time = end_process_block - start_process_block
    
    return result/matrix_block_num, process_block_time, 0

# 实验函数配置
STEP_MAP = {
    "estimate_svd_weights_and_process": {"loss_threshold": 1.0, "step_func": estimate_svd_weights_and_process},
    "step_func_default": {"loss_threshold": 1.0, "step_func": step_default},
    "step_func_row_block": {"loss_threshold": 1.0, "step_func": step_row_block},
    "step_row_block_spectral_normalization": {"loss_threshold": 1.0, "step_func": step_row_block_spectral_normalization},
    "step_func_column_block": {"loss_threshold": 1.0, "step_func": step_column_block}, 
    "step_func_quadrant_block": {"loss_threshold": 1.0, "step_func": step_block_matrix_flexible},
}

MUON_BLOCK_MATRIX_EXPERIMENT_DIR = os.getenv("MUON_BLOCK_MATRIX_EXPERIMENT_DIR")
RESULTS_BASE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Results")
BASE_LOG_PATH = os.path.join(RESULTS_BASE, "Logs/MuonBlockMatrix")

def singular_value_range(matrix):
    """计算矩阵的奇异值范围"""
    # 直接在GPU上计算奇异值
    _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    min_sv = singular_values.min().item()
    max_sv = singular_values.max().item()
    return min_sv, max_sv

# 创建日志目录
log_dir = os.path.join(BASE_LOG_PATH, "multiple_functions_experiment")
os.makedirs(log_dir, exist_ok=True)

# 设置日志文件
log_file = os.path.join(log_dir, f"multi_func_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {message}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 对于STEP_MAP中的每个函数
for func_name, func_info in STEP_MAP.items():
    step_func = func_info["step_func"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing function: {func_name}")
    logger.info(f"{'='*60}")
    
    # 每个函数处理20个矩阵
    for matrix_idx in range(10):
        m = 2048
        n = 1024

        matrix = torch.randn(m, n, device=device, dtype=torch.float32)*5
        
        # 计算原始矩阵的奇异值范围
        min_sv_orig, max_sv_orig = singular_value_range(matrix)
        
        # 用当前函数处理矩阵
        try:
            # 调用函数，注意参数格式统一
            processed_matrix,_,_ = step_func(matrix, steps=5, matrix_block_num=16)
            
            # 将处理后的矩阵转换为Float32用于SVD计算
            processed_matrix_f32 = processed_matrix.to(torch.float32)
            
            # 计算处理后的矩阵的奇异值范围
            min_sv_proc, max_sv_proc = singular_value_range(processed_matrix_f32)
            
            # 记录结果
            logger.info(f"Matrix {matrix_idx+1}: size={m}x{n}, function={func_name}")
            logger.info(f"  Original: sv_range=[{min_sv_orig:.5f}, {max_sv_orig:.5f}]")
            logger.info(f"  Processed: sv_range=[{min_sv_proc:.5f}, {max_sv_proc:.5f}]")
            if min_sv_orig != 0 and min_sv_proc != 0:
                logger.info(f"  Ratio (max/min): orig={max_sv_orig/min_sv_orig:.2f}, proc={max_sv_proc/min_sv_proc:.2f}")
            
        except Exception as e:
            logger.info(f"Matrix {matrix_idx+1}: Error processing with {func_name} - {e}")

logger.info("\nAll tests completed.")