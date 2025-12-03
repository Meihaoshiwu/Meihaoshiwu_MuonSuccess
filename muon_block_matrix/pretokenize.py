# preprocess.py
import os
from .data import ExperimentPreparer, load_dataset_by_name
from .config import TOKENIZED_CACHE, DATASET_CACHE, MODEL_CACHE, DATA_EXTRACTED_DIR

def create_sharded_directories(input_dir, num_shards=32):
    """
    将输入目录的文件分配到多个分片目录（创建符号链接）
    返回分片目录列表
    """
    import os
    import glob
    import hashlib
    
    # 获取所有txt文件
    all_files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"📁 找到 {len(all_files)} 个txt文件")
    
    # 创建分片目录
    base_shard_dir = os.path.join(DATASET_CACHE, "openwebtext_shards")
    os.makedirs(base_shard_dir, exist_ok=True)
    
    shard_dirs = []
    for i in range(num_shards):
        shard_dir = os.path.join(base_shard_dir, f"shard_{i}")
        shard_dirs.append(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)
    
    # 将文件分配到不同分片目录（创建符号链接）
    print(f"🔄 将文件分配到 {num_shards} 个分片...")
    
    files_per_shard = {}
    for file_path in all_files:
        # 使用文件名哈希决定分片
        filename = os.path.basename(file_path)
        
        # 从文件名中提取数字部分，如 doc_020609.txt
        try:
            import re
            match = re.search(r'\d+', filename)
            if match:
                file_num = int(match.group())
                shard_idx = file_num % num_shards
            else:
                # 如果没有数字，用哈希
                file_hash = int(hashlib.md5(filename.encode()).hexdigest(), 16)
                shard_idx = file_hash % num_shards
        except:
            # 如果解析失败，使用简单哈希
            file_hash = int(hashlib.md5(filename.encode()).hexdigest(), 16)
            shard_idx = file_hash % num_shards
        
        # 创建符号链接
        link_path = os.path.join(shard_dirs[shard_idx], filename)
        if not os.path.exists(link_path):
            os.symlink(os.path.abspath(file_path), link_path)
        
        # 统计
        files_per_shard[shard_idx] = files_per_shard.get(shard_idx, 0) + 1
    
    # 打印统计信息
    for shard_idx in range(num_shards):
        count = files_per_shard.get(shard_idx, 0)
        print(f"  分片 {shard_idx}: {count} 个文件")
    
    print(f"✅ 文件分配完成")
    return shard_dirs

def cleanup_sharded_directories(shard_dirs):
    """清理分片目录和符号链接"""
    print("🔄 清理分片目录...")
    for shard_dir in shard_dirs:
        if os.path.exists(shard_dir):
            # 删除所有符号链接
            for filename in os.listdir(shard_dir):
                link_path = os.path.join(shard_dir, filename)
                os.unlink(link_path)
            # 删除目录
            os.rmdir(shard_dir)
    print("✅ 清理完成")

def preprocess_dataset(dataset_name: str, tokenizer_name: str = "Qwen/Qwen2.5-0.5B"):
    """独立的预处理函数"""
    # 定义输出文件路径
    output_file = os.path.join(TOKENIZED_CACHE, f"{dataset_name}.bin")
    print(f"🔄 开始预处理数据集: {dataset_name}, output_file = {output_file}")
    
    # 如果已经预处理过，直接返回
    if os.path.exists(output_file):
        print(f"✅ 预处理文件已存在: {output_file}")
        return output_file
    
    # 对于 openwebtext-local_txt，使用分片目录方案
    if dataset_name == "openwebtext-local_txt":
        output_dir = os.path.join(DATA_EXTRACTED_DIR, "openwebtext")
        
        print("🔧 使用分片目录方案处理 openwebtext-local_txt")
        
        # 1. 创建分片目录
        num_shards = 32  # 根据CPU核心数调整
        shard_dirs = create_sharded_directories(output_dir, num_shards=num_shards)
        
        try:
            # 2. 创建preparer，传入分片目录
            preparer = ExperimentPreparer(
                texts=[],  # 传递空列表，因为使用分片目录
                tokenizer_name=tokenizer_name,
                model_cache_dir=MODEL_CACHE,
                output_file=output_file,
                batch_size=64,  # 可以调整
                shard_dirs=shard_dirs,  # 传递分片目录
            )
            
            # 3. 执行tokenize
            preparer.tokenize()
            
            print(f"✅ 预处理完成: {output_file}")
            return output_file
            
        finally:
            # 4. 清理分片目录
            cleanup_sharded_directories(shard_dirs)
    
    else:
        # 其他数据集使用原有逻辑
        print("🔧 使用原有逻辑处理数据集")
        
        # 加载原始数据集
        raw_dataset = load_dataset_by_name(dataset_name)
        if "train" in raw_dataset:
            texts = raw_dataset["train"]["text"]
        else:
            texts = raw_dataset["text"]
        
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