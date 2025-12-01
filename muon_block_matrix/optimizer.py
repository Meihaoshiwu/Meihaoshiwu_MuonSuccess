import math
import torch
import os

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

def step_default(G, steps):
    return zeropower_via_newtonschulz5(G, steps)

def process_block(blocked_matrix, steps):
    if torch.norm(blocked_matrix) > 1e-7:
        orthogonalized_block = zeropower_via_newtonschulz5(blocked_matrix, steps)
        return orthogonalized_block
    else:
        return blocked_matrix

def step_column_block(G, steps):
    """
    将矩阵的列分成4块，每块独立进行正交化处理
    """
    result = torch.zeros_like(G) # 用于保存结果
    cols = G.shape[1]  # 总列数
    block_size = cols // 4  # 每块的列数
    
    # 处理前3个完整的块
    for i in range(3):
        start_col = i * block_size
        end_col = (i + 1) * block_size
        column_block = G[:, start_col:end_col] # 取所有行，[start_col,end_col)列

        result[:, start_col:end_col] = process_block(column_block, steps)
    
    # 处理最后一块（可能包含剩余的列）
    start_col = 3 * block_size
    last_block = G[:, start_col:]
    result[:, start_col:] = process_block(last_block, steps)
    
    return result

def step_row_block(G, steps):
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
        result[start_row:end_row, :] = process_block(row_block, steps)
    
    # 处理最后一块（可能包含剩余的行）
    start_row = 3 * block_size
    last_block = G[start_row:, :]
    result[start_row:, :] = process_block(last_block, steps)
    
    return result

def step_quadrant_block(G, steps):
    """
    将矩阵分成2x2的分成四块
    """
    result = torch.zeros_like(G)
    
    rowidx = G.shape[0]//2
    colidx = G.shape[1]//2

    G11 = G[:rowidx, :colidx]
    result[:rowidx, :colidx] = process_block(G11, steps)
    G12 = G[:rowidx, colidx:]
    result[:rowidx, colidx:] = process_block(G12, steps)
    G21 = G[rowidx:, :colidx]
    result[rowidx:, :colidx] = process_block(G21, steps)
    G22 = G[rowidx:, colidx:]
    result[rowidx:, colidx:] = process_block(G22, steps)
    
    return result

# 实验函数配置
STEP_MAP = {
    "step_func_default": {"loss_threshold": 1.0, "step_func": step_default},
    "step_func_column_block": {"loss_threshold": 1.0, "step_func": step_column_block}, 
    "step_func_row_block": {"loss_threshold": 1.0, "step_func": step_row_block},
    "step_func_quadrant_block": {"loss_threshold": 1.0, "step_func": step_quadrant_block}
}

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
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
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
                u = self.step_func(g, steps=group["ns_steps"])

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