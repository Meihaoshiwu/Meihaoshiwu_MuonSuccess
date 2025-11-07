# preprocess.py
import os
from .data import ExperimentPreparer, load_dataset_by_name
from .config import TOKENIZED_CACHE, DATASET_CACHE, MODEL_CACHE

def preprocess_dataset(dataset_name: str, tokenizer_name: str = "Qwen/Qwen2.5-0.5B"):
    """独立的预处理函数"""
    # 定义输出文件路径
    output_file = os.path.join(TOKENIZED_CACHE, f"{dataset_name}.bin")
    print(f"🔄 开始预处理数据集: {dataset_name}, output_file = {output_file}, DATASET_CACHE = {DATASET_CACHE}")
    # 加载原始数据集
    raw_dataset = load_dataset_by_name(dataset_name)
    if "train" in raw_dataset:
        texts = raw_dataset["train"]["text"]
    else:
        texts = raw_dataset["text"] 
    
    # 如果已经预处理过，直接返回
    if os.path.exists(output_file):
        print(f"✅ 预处理文件已存在: {output_file}")
        return output_file
    
    # 执行 tokenize
    preparer = ExperimentPreparer(
        texts=texts,
        tokenizer_name=tokenizer_name,
        model_cache_dir=MODEL_CACHE,
        output_file=output_file,
    )
    preparer.tokenize()
    
    print(f"✅ 预处理完成: {output_file}")
    return output_file

if __name__ == "__main__":
    # 命令行接口
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B", help="tokenizer名称")
    args = parser.parse_args()
    preprocess_dataset(args.dataset, args.tokenizer)