import argparse
import os
from loguru import logger

from .config import *
from .train import run_experiment
from .optimizer import STEP_MAP, step_default

def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_func_name", type=str, default="default")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="muon")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=512) # 一个词使用多长的向量来表示（词向量维数）
    parser.add_argument("--max_position_embeddings", type=int, default=2048) # 最长给多少个词编码
    parser.add_argument("--max_length", type=int, default=512)
    # 每个样本有多少个token（即词向量）这决定了局部连续文本长度，影响学习效率，如果文本太短很难学习到词语之间联系
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss_threshold", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=10000000000, help="最大训练token数量")
    args = parser.parse_args()

    timestamp = get_timestamp()
    base_log_path = os.path.join(RESULTS_BASE, "Logs/MuonBlockMatrix")
    experiment_dir = f"{base_log_path}/experiment_{timestamp}"

    experiment_config = ExperimentConfig(
        step_func_name="default",
        step_func=step_default,
        log_file_path=experiment_dir,
        model_name=args.model,
        dataset_name=args.dataset,
        optimizer_name=args.optimizer,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        loss_threshold=args.loss_threshold,
        max_epochs=args.max_epochs,
        lr=args.lr,
        wd=args.wd,
        max_tokens=args.max_tokens,
    )

    print(f"🧪 开始自动化实验序列")
    print(f"使用模型缓存: {MODEL_CACHE}")
    print(f"使用数据集缓存: {DATASET_CACHE}")
    print(f"使用分词缓存: {TOKENIZED_CACHE}")
    
    results = {}
    for i, (func_name, config_dict) in enumerate(STEP_MAP.items()):
        # 运行实验
        experiment_config.step_func_name = func_name
        experiment_config.step_func = config_dict.get("step_func")
        final_loss, all_losses = run_experiment(experiment_config)
        
        # 保存结果
        # results[func_name] = {
        #     'final_loss': final_loss,
        #     'all_losses': all_losses,
        #     'converged': final_loss < experiment_config.loss_threshold
        # }
        
        # 为下一个实验等待一下，确保资源释放
        import time
        time.sleep(5)
    
    # for exp_name, result in results.items():
    #     status = "✅ 收敛" if result['converged'] else "❌ 未收敛"
    #     print(f"{exp_name}: 最终损失 = {result['final_loss']:.4f} {status}")

if __name__ == "__main__":
    main()