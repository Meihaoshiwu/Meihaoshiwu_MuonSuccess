import os
import matplotlib.pyplot as plt
import re
import sys

def extract_data_from_log(log_file_path):
    """从日志文件提取奇异值数据"""
    data = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        line = lines[i].strip()
        
        # 查找包含"sv_range="的行
        if 'sv_range=' in line:
            # 提取奇异值范围
            match = re.search(r'sv_range=\[([-\d\.e+-]+),\s*([-\d\.e+-]+)\]', line)
            
            if match:
                # 区分是原始矩阵还是处理后矩阵
                if 'Original:' in line:
                    # 这是原始矩阵的奇异值范围
                    # 需要先找到对应的矩阵信息行
                    for j in range(i-1, max(i-5, -1), -1):
                        if 'Matrix' in lines[j] and 'size=' in lines[j]:
                            matrix_match = re.search(r'Matrix\s+(\d+):', lines[j])
                            if matrix_match:
                                matrix_index = int(matrix_match.group(1))
                                
                                # 提取原始矩阵奇异值
                                orig_min = float(match.group(1))
                                orig_max = float(match.group(2))
                                
                                # 查找处理后的奇异值（应该在下一行）
                                if i+1 < len(lines) and 'Processed:' in lines[i+1]:
                                    proc_match = re.search(r'sv_range=\[([-\d\.e+-]+),\s*([-\d\.e+-]+)\]', lines[i+1])
                                    if proc_match:
                                        proc_min = float(proc_match.group(1))
                                        proc_max = float(proc_match.group(2))
                                        
                                        data.append({
                                            'index': matrix_index,
                                            'orig_min': orig_min,
                                            'orig_max': orig_max,
                                            'proc_min': proc_min,
                                            'proc_max': proc_max
                                        })
                                break
    
    return data

def plot_singular_value_ranges(data, log_file_path):
    """绘制奇异值范围图"""
    indices = [d['index'] for d in data]
    
    # 提取数据
    orig_mins = [d['orig_min'] for d in data]
    orig_maxs = [d['orig_max'] for d in data]
    proc_mins = [d['proc_min'] for d in data]
    proc_maxs = [d['proc_max'] for d in data]
    
    plt.figure(figsize=(12, 8))
    
    # 绘制四条线，使用不同颜色，都是实线
    plt.plot(indices, orig_maxs, 'r-', linewidth=1.5, label='Original Max')
    plt.plot(indices, orig_mins, 'g-', linewidth=1.5, label='Original Min')
    plt.plot(indices, proc_maxs, 'b-', linewidth=1.5, label='Processed Max')
    plt.plot(indices, proc_mins, 'orange', linewidth=1.5, label='Processed Min')
    
    plt.xlabel('Matrix Index')
    plt.ylabel('Singular Value')
    plt.title(f'Singular Value Ranges for {len(data)} Matrices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 去掉了对数坐标设置
    
    # 从日志文件路径生成图片文件路径
    plot_file = log_file_path.replace('.log', '.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_file}")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python plot_utils.py <日志文件路径>")
        return
    
    log_file_path = sys.argv[1]
    print(log_file_path)
    data = extract_data_from_log(log_file_path)
    
    if data:
        plot_singular_value_ranges(data, log_file_path)
    else:
        print("未从日志文件中提取到数据")

if __name__ == "__main__":
    main()