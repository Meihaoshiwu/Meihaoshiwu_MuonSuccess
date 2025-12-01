#!/usr/bin/env python3
import os
import argparse
from transformers import Qwen2Tokenizer, LlamaTokenizer
from huggingface_hub import snapshot_download
import requests

from datasets import load_dataset
from loguru import logger
from .config import MODEL_CACHE, DATASET_CACHE, DATASET_PATH

# 修改为镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def pre_download_resources(dataset_names, model_name, use_snapshot_for=None):
    """预下载所有需要的资源
    
    Args:
        dataset_names: 要下载的数据集名称列表
        model_name: 模型名称
        use_snapshot_for: 使用snapshot方式下载的数据集名称列表
    """
    
    # 确保目录存在
    os.makedirs(MODEL_CACHE, exist_ok=True)
    os.makedirs(DATASET_CACHE, exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    logger.info("🔽 开始预下载资源...")
    
    # 下载Tokenizer
    tokenizer_success = _download_tokenizer(model_name)
    
    # 下载数据集
    dataset_success = True
    if dataset_names:
        dataset_success = _download_datasets(dataset_names, use_snapshot_for or [])
    
    if tokenizer_success and dataset_success:
        logger.info("🎉 所有资源预下载完成！")
        logger.info(f"模型缓存: {MODEL_CACHE}")
        logger.info(f"数据集路径: {DATASET_PATH}")
        return True
    else:
        logger.error("❌ 资源下载失败，请检查网络连接或配置")
        return False

def _download_tokenizer(model_name):
    """下载模型对应的tokenizer"""
    logger.info(f"下载Tokenizer: {model_name}")
    
    try:
        if model_name.startswith("qwen"):
            tokenizer_name = "Qwen/Qwen2.5-0.5B"
            tokenizer_class = Qwen2Tokenizer
        elif model_name.startswith("llama"):
            tokenizer_name = "huggyllama/llama-7b"
            tokenizer_class = LlamaTokenizer
        else:
            logger.error(f"❌ 不支持的模型类型: {model_name}")
            return False
        
        # 首先尝试从缓存加载
        try:
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                cache_dir=MODEL_CACHE,
                local_files_only=True
            )
            logger.info("✅ Tokenizer已存在于缓存中")
            return True
        except (OSError, Exception) as e:
            logger.warning(f"⚠️  缓存中未找到Tokenizer，开始下载: {e}")
            
        # 下载Tokenizer
        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            cache_dir=MODEL_CACHE,
            trust_remote_code=True  # 对于某些模型可能需要这个参数
        )
        logger.info(f"✅ Tokenizer已下载到: {MODEL_CACHE}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 下载Tokenizer失败: {e}")
        logger.info("💡 建议检查：")
        logger.info("   1. 网络连接是否正常")
        logger.info("   2. 模型名称是否正确")
        logger.info("   3. 是否有访问该模型的权限")
        return False

def _download_datasets(dataset_names, use_snapshot_for):
    """下载指定的数据集
    
    Args:
        dataset_names: 要下载的数据集名称列表
        use_snapshot_for: 使用snapshot方式下载的数据集名称列表
    """
    success = True
    
    for dataset_name in dataset_names:
        logger.info(f"下载数据集: {dataset_name}")
        
        # 数据集名称到仓库ID的映射
        name2repo = {
            "openwebtext-100k": "Elriggs/openwebtext-100k",
            "openwebtext": "Skylion007/openwebtext",
            "wikitext-103": "wikitext",
        }
        
        if dataset_name in name2repo:
            repo_id = name2repo[dataset_name]
            
            try:
                # 检查是否应该使用 snapshot 方式下载
                if dataset_name in use_snapshot_for:
                    dataset_success = _download_dataset_snapshot(repo_id, dataset_name)
                else:
                    # 对于 OpenWebText 数据集，默认使用 snapshot download
                    if dataset_name in ["openwebtext-100k", "openwebtext"]:
                        logger.info(f"⚠️  检测到 {dataset_name} 可能不兼容，建议使用 --use_snapshot 参数")
                        dataset_success = _download_dataset_standard(repo_id, dataset_name)
                    else:
                        dataset_success = _download_dataset_standard(repo_id, dataset_name)
                        
                if not dataset_success:
                    success = False
                        
            except Exception as e:
                logger.error(f"❌ 下载数据集 '{dataset_name}' 失败: {e}")
                success = False
        else:
            logger.warning(f"⚠️  未知数据集: {dataset_name}，跳过下载")
            success = False
    
    return success

def _download_dataset_snapshot(repo_id, dataset_name):
    """使用 snapshot download 方式下载数据集"""
    try:
        # 创建数据集特定目录
        dataset_dir = os.path.join(DATASET_PATH, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        logger.info(f"使用 snapshot download 方式下载 {repo_id}")
        logger.info(f"数据集将保存到: {dataset_dir}")
        
        # 下载数据集快照
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dataset_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        logger.info(f"✅ 数据集 '{dataset_name}' 快照下载完成")
        
        # 检查下载的文件
        downloaded_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, dataset_dir)
                downloaded_files.append(relative_path)
        
        if downloaded_files:
            logger.info(f"下载的文件列表:")
            for file in downloaded_files[:10]:  # 只显示前10个文件
                logger.info(f"  - {file}")
            if len(downloaded_files) > 10:
                logger.info(f"  ... 还有 {len(downloaded_files) - 10} 个文件")
        else:
            logger.warning(f"⚠️  数据集目录为空，可能没有下载到数据文件")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ 快照下载数据集 '{dataset_name}' 失败: {e}")
        logger.info("💡 可以尝试使用标准下载方式，或检查数据集名称是否正确")
        return False

def _download_dataset_standard(repo_id, dataset_name):
    """使用标准方式下载数据集"""
    try:
        if repo_id == "wikitext":
            dataset = load_dataset(
                repo_id,
                "wikitext-103-v1",
                cache_dir=DATASET_CACHE
            )
        else:
            dataset = load_dataset(
                repo_id,
                cache_dir=DATASET_CACHE
            )
        
        logger.info(f"✅ 数据集 '{dataset_name}' 已准备就绪")
        if hasattr(dataset, 'get') and 'train' in dataset:
            try:
                if hasattr(dataset['train'], '__len__'):
                    logger.info(f"   训练集大小: {len(dataset['train'])} 条样本")
                else:
                    logger.info("   训练集: 流式数据集（大小未知）")
            except:
                logger.info("   训练集: 大小信息不可用")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ 下载数据集 '{dataset_name}' 失败: {e}")
        
        # 如果是数据集脚本不兼容的错误，建议使用 snapshot 方式
        if "dataset script" in str(e).lower() or ".py" in str(e):
            logger.info("💡 这个数据集使用了旧的.py脚本格式，不再被支持")
            logger.info("   建议使用 --use_snapshot 参数重新下载")
            
        return False

def check_network_connection():
    """检查网络连接"""
    try:
        response = requests.get("https://hf-mirror.com", timeout=10)
        return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="预下载训练所需资源")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["openwebtext-100k"],
        help="要下载的数据集名称，支持: openwebtext-100k, openwebtext, wikitext-103"
    )
    parser.add_argument(
        "--model_name",
        default="qwen",
        choices=["qwen", "llama"],
        help="模型类型"
    )
    parser.add_argument(
        "--use_snapshot",
        nargs="*",
        default=None,
        help="使用snapshot方式下载的数据集名称，不指定则不使用snapshot方式"
    )
    parser.add_argument(
        "--skip_network_check",
        action="store_true",
        help="跳过网络连接检查"
    )
    
    args = parser.parse_args()
    
    # 检查网络连接
    if not args.skip_network_check and not check_network_connection():
        logger.error("❌ 网络连接失败，请检查网络设置")
        logger.info("💡 可以尝试：")
        logger.info("   1. 检查网络连接")
        logger.info("   2. 使用 --skip_network_check 跳过检查")
        logger.info("   3. 配置代理或更换镜像源")
        return
    
    success = pre_download_resources(
        dataset_names=args.datasets,
        model_name=args.model_name,
        use_snapshot_for=args.use_snapshot
    )
    
    if not success:
        logger.error("❌ 资源下载失败")
        exit(1)

if __name__ == "__main__":
    main()