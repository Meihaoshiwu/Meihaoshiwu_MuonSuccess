import math
import torch
import os
from loguru import logger
# ---- 牛顿-舒尔茨正交化 ----
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'  # 限制编译线程数量，避免编译线程抢占太多资源影响进程通信
@torch.compile
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
    return zeropower_via_newtonschulz5(G, steps)

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
        return process_block(G, steps)
    
    block_size = cols // matrix_block_num  # 每块的列数

    for i in range(matrix_block_num-1):
        start_col = i * block_size
        end_col = (i + 1) * block_size
        column_block = G[:, start_col:end_col] # 取所有行，[start_col,end_col)列
        result[:, start_col:end_col] = process_block(column_block, steps)
    
    # 处理最后一块（可能包含剩余的列）
    start_col = (matrix_block_num-1) * block_size
    last_block = G[:, start_col:]
    result[:, start_col:] = process_block(last_block, steps)
    
    return result

def step_row_block(G, steps, matrix_block_num: int):
    """
    将矩阵的行分成matrix_block_num块，每块独立进行正交化处理
    """
    result = torch.zeros_like(G)
    rows = G.shape[0]  # 总行数
    
    # 检查是否支持这么多分块
    if matrix_block_num > rows:
        logger.info(f"Warning: 矩阵只有{rows}行，但要求分成{matrix_block_num}块，将作为整体处理")
        return process_block(G, steps)
    
    block_size = rows // matrix_block_num  # 每块的行数
    
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
    
    return result

# 横纵都切分
def step_block_matrix_flexible(G, steps, matrix_block_num: int):
    """
    更灵活的分块函数，可以处理非平方数的分块数
    例如：block_num=12 -> 可能会分成3x4或4x3等
    """
    result = torch.zeros_like(G)
    rows, cols = G.shape
    
    # 寻找最接近平方根的两个因数
    row_blocks, col_blocks = -1, -1
    sqrt_block = int(math.sqrt(matrix_block_num))
    for i in range(sqrt_block, 1, -1): # 至少不退化到列分块
        if matrix_block_num % i == 0:
            row_blocks, col_blocks = i, matrix_block_num // i
            break
    
    if row_blocks < 0:
        logger.info(f"Warning: 无法将{matrix_block_num}分解为合适的因数，将作为整体处理")
        return process_block(G, steps)

    if rows // row_blocks < 1 or cols // col_blocks < 1:
        logger.info(f"Warning: 分块后某些块没有元素，将作为整体处理")
        return process_block(G, steps)
    
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
    
    return result

# 轮流处理某一个分块,其他块置零
def step_row_block_process_one(G, steps, matrix_block_num: int):
    """
    将矩阵的行分成matrix_block_num块，每块独立进行正交化处理
    """
    rows = G.shape[0]  # 总行数
    
    # 矩阵太小没必要分块
    if matrix_block_num*matrix_block_num > rows:
        logger.info(f"Warning: 矩阵{rows}行，要求分成{matrix_block_num}块，将作为整体处理")
        return process_block(G, steps)

    # 初始化静态变量（函数属性）
    if not hasattr(step_row_block_process_one, 'current_block'):
        step_row_block_process_one.current_block = 0  # 当前要处理的块索引

    result = torch.zeros_like(G)
    
    block_size = rows // matrix_block_num  # 每块的行数

    if step_row_block_process_one.current_block < matrix_block_num-1:
        start_row = step_row_block_process_one.current_block * block_size
        end_row = (step_row_block_process_one.current_block + 1) * block_size
        row_block = G[start_row:end_row, :]
        result[start_row:end_row, :] = process_block(row_block, steps)
    else:
        # 处理最后一块（可能包含剩余的行）
        start_row = (matrix_block_num-1) * block_size
        last_block = G[start_row:, :]
        result[start_row:, :] = process_block(last_block, steps)
    
    # 更新块索引，准备处理下一块
    step_row_block_process_one.current_block = (step_row_block_process_one.current_block + 1) % matrix_block_num
    
    return result

# 随机处理其中某一个矩阵
def step_row_random_process_one_block(G, steps, matrix_block_num: int):
    """
    随机处理其中一个矩阵块，其他块置零
    """
    rows = G.shape[0]  # 总行数
    
    # 矩阵太小没必要分块
    if matrix_block_num*matrix_block_num > rows:
        logger.info(f"Warning: 矩阵{rows}行，要求分成{matrix_block_num}块，将作为整体处理")
        return process_block(G, steps)

    result = torch.zeros_like(G)
    
    # 随机选择一个块索引
    block_idx = torch.randint(0, matrix_block_num, (1,)).item()
    
    # 计算块大小
    block_size = rows // matrix_block_num  # 每块的行数
    
    # 处理选中的块
    if block_idx < matrix_block_num-1:
        start_row = block_idx * block_size
        end_row = (block_idx + 1) * block_size
        block_data = G[start_row:end_row, :]
        result[start_row:end_row, :] = process_block(block_data, steps)
    else:
        # 处理最后一块（可能包含剩余的行）
        start_row = (matrix_block_num-1) * block_size
        block_data = G[start_row:, :]
        result[start_row:, :] = process_block(block_data, steps)
    
    return result

def estimate_svd_weights_and_process(G, steps, matrix_block_num: int):
    """
    对原始矩阵G分块估计奇异值之和，得到权重
    然后将权重乘到处理之后的矩阵上
    """
    rows = G.shape[0]  # 总行数
    
    # 矩阵太小没必要分块
    if matrix_block_num*matrix_block_num > rows:
        logger.info(f"Warning: 矩阵{rows}行，要求分成{matrix_block_num}块，将作为整体处理")
        return process_block(G, steps)

    block_size = rows // matrix_block_num
    num_samples=20

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
    for block in original_blocks:
        sv_est = randomized_nuclear_norm_estimate_fast(block, num_samples)
        sv_estimates.append(sv_est)
    
    # 3. 计算权重（每个块的奇异值之和占总和的比例）
    sv_estimates_tensor = torch.tensor(sv_estimates, device=G.device, dtype=G.dtype)
    # 检查奇异值之和是否为零，避免除零错误
    if torch.sum(sv_estimates_tensor) != 0:
        weights = sv_estimates_tensor / torch.sum(sv_estimates_tensor)
    else:
        weights = torch.ones_like(sv_estimates_tensor) / len(sv_estimates_tensor)
        logger.error("警告：所有块的奇异值估计都为零，使用均匀权重")
    
    # 4. 再次处理每个块，但这次使用原始矩阵计算好的权重
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
    
    return result

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

# 谱范数归一化，分块正交之后乘以1/4
def step_row_block_quarter(G, steps):
    """
    将矩阵的行分成4块，每块独立进行正交化处理
    """
    result = torch.zeros_like(G)
    rows = G.shape[0]  # 总行数
    block_size = rows // 4  # 每块的行数
    
    # 处理前3个完整的块
    for i in range(3):
        start_row = i * block_size
        end_row = (i + 1) * block_size
        row_block = G[start_row:end_row, :]
        result[start_row:end_row, :] = 1/4*process_block(row_block, steps)
    
    # 处理最后一块（可能包含剩余的行）
    start_row = 3 * block_size
    last_block = G[start_row:, :]
    result[start_row:, :] = 1/4*process_block(last_block, steps)
    
    return result

# 实验函数配置
STEP_MAP = {
    "estimate_svd_weights_and_process": {"loss_threshold": 1.0, "step_func": estimate_svd_weights_and_process},
    "step_func_default": {"loss_threshold": 1.0, "step_func": step_default},
    "step_func_row_block": {"loss_threshold": 1.0, "step_func": step_row_block},
    "step_row_block_process_one": {"loss_threshold": 1.0, "step_func": step_row_block_process_one},
    "step_row_block_quarter": {"loss_threshold": 1.0, "step_func": step_row_block_quarter},
    "step_row_random_process_one_block": {"loss_threshold": 1.0, "step_func": step_row_random_process_one_block},
    "step_func_column_block": {"loss_threshold": 1.0, "step_func": step_column_block}, 
    "step_func_quadrant_block": {"loss_threshold": 1.0, "step_func": step_block_matrix_flexible}
}

MATRIX_BLOCK_NUM=16
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        step_func:callable,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.9),
        adamw_eps=1e-8,
        matrix_block_num=MATRIX_BLOCK_NUM
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        self.step_func = step_func
        self.matrix_block_num=matrix_block_num
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = self.step_func(g, steps=group["ns_steps"], matrix_block_num=self.matrix_block_num)

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss

def get_optimizer(step_func:callable, optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            step_func,
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"