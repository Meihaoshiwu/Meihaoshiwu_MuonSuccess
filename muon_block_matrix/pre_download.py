#!/usr/bin/env python3
import os
import argparse
from transformers import Qwen2Tokenizer
from datasets import load_dataset
from loguru import logger
from .config import MODEL_CACHE, DATASET_CACHE, DATASET_PATH

def pre_download_resources(dataset_names=None, tokenizer_name="Qwen/Qwen2.5-0.5B"):
    """预下载所有需要的资源"""
    
    # 确保目录存在
    os.makedirs(MODEL_CACHE, exist_ok=True)
    os.makedirs(DATASET_CACHE, exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    logger.info("🔽 开始预下载资源...")
    
    # 1. 下载Tokenizer
    logger.info(f"下载Tokenizer: {tokenizer_name}")
    try:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=MODEL_CACHE,
            local_files_only=True  # 先检查本地是否已有
        )
        logger.info("✅ Tokenizer已存在于缓存中")
    except (OSError, TypeError):
        # 本地没有，需要下载
        tokenizer = Qwen2Tokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=MODEL_CACHE
        )
        logger.info(f"✅ Tokenizer已下载到: {MODEL_CACHE}")
    
    # 2. 下载数据集
    if dataset_names:
        for dataset_name in dataset_names:
            logger.info(f"下载数据集: {dataset_name}")
            
            # 数据集名称到路径的映射
            name2path = {
                "openwebtext-100k": "Elriggs/openwebtext-100k",
                "openwebtext": "Skylion007/openwebtext",
                "wikitext-103": "wikitext",
            }
            
            if dataset_name in name2path:
                try:
                    # 尝试加载数据集（如果已缓存则不会重复下载）
                    dataset = load_dataset(
                        name2path[dataset_name],
                        cache_dir=DATASET_CACHE
                    )
                    logger.info(f"✅ 数据集 '{dataset_name}' 已准备就绪")
                    logger.info(f"   训练集大小: {len(dataset['train'])} 条样本")
                except Exception as e:
                    logger.error(f"❌ 下载数据集 '{dataset_name}' 失败: {e}")
            else:
                logger.warning(f"⚠️  未知数据集: {dataset_name}，跳过下载")
    
    logger.info("🎉 所有资源预下载完成！")
    logger.info(f"模型缓存: {MODEL_CACHE}")
    logger.info(f"数据集缓存: {DATASET_CACHE}")

def main():
    parser = argparse.ArgumentParser(description="预下载训练所需资源")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["openwebtext-100k"],
        help="要下载的数据集名称，支持: openwebtext-100k, openwebtext, wikitext-103"
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-0.5B",
        help="Tokenizer名称"
    )
    
    args = parser.parse_args()
    
    pre_download_resources(
        dataset_names=args.datasets,
        tokenizer_name=args.tokenizer
    )

if __name__ == "__main__":
    main()