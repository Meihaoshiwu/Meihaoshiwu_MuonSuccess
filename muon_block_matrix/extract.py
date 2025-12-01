import os
import tarfile
import lzma
import re

from .config import DATASET_PATH

def extract_dataset(dataset_name="openwebtext"):
    """手动解压 OpenWebText 数据集"""
    
    # 构建输出目录：DATASET_PATH/output/数据集名称/
    DATA_DIR = os.path.join(DATASET_PATH, "openwebtext/subsets")
    OUTPUT_DIR = os.path.join(DATASET_PATH, "openwebtext/output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    # 找到所有的 tar 文件
    tar_files = [os.path.join(DATA_DIR, f"urlsf_subset{i:02d}.tar") for i in range(21)]
    
    total_files = 0
    for tar_path in tar_files:
        if not os.path.exists(tar_path):
            print(f"跳过不存在的文件: {tar_path}")
            continue
            
        print(f"处理文件: {tar_path}")
        
        # 解压 tar 文件
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if member.name.endswith('.xz'):
                    # 提取 xz 文件
                    xz_file = tar.extractfile(member)
                    if xz_file:
                        # 解压 xz 并读取内容
                        try:
                            with lzma.open(xz_file, 'rt', encoding='utf-8') as f:
                                text_content = f.read()
                            
                            # 清理文本（与原脚本相同的处理）
                            cleaned_text = re.sub("\n\n\n+", "\n\n", text_content).strip()
                            
                            # 保存为 txt 文件
                            output_filename = os.path.join(
                                OUTPUT_DIR, 
                                f"doc_{total_files:06d}.txt"
                            )
                            with open(output_filename, 'w', encoding='utf-8') as out_file:
                                out_file.write(cleaned_text)
                            
                            total_files += 1
                            if total_files % 1000 == 0:
                                print(f"已处理 {total_files} 个文档...")
                                
                        except Exception as e:
                            print(f"处理文件 {member.name} 时出错: {e}")
                            continue
    
    print(f"解压完成！共提取 {total_files} 个文档到 {OUTPUT_DIR}")
    return OUTPUT_DIR, total_files

if __name__ == "__main__":
    # 可以指定数据集名称，默认为 "openwebtext"
    output_dir, file_count = extract_dataset("openwebtext")
    print(f"数据集已保存到: {output_dir}")
    print(f"总共生成 {file_count} 个文本文件")