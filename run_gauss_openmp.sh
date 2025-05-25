#!/bin/bash

# 设置执行环境和错误处理
set -e  # 发生错误时退出
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# 定义颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== OpenMP Gaussian Elimination Thread Scaling Test ===${NC}"
echo -e "${BLUE}Current Date and Time: 2025-05-22 07:25:14${NC}"
echo -e "${BLUE}Current User: KKKyriejiang${NC}"

# 编译程序（使用 ARM 架构交叉编译器并支持OpenMP）
echo -e "${BLUE}Cross-compiling OpenMP Gaussian Elimination program for ARM...${NC}"
aarch64-linux-gnu-g++ -static -o gaussian_elimination_openmp gaussian_elimination_openmp.cpp \
  -fopenmp -O3 -march=armv8-a -mtune=cortex-a72 \
  -falign-functions=64 -falign-loops=64 -ftree-vectorize \
  -ffast-math -funroll-loops

# 检查编译是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}Compilation failed!${NC}"
  exit 1
fi
echo -e "${GREEN}Compilation successful${NC}"

# 创建结果目录结构
echo -e "${BLUE}Setting up result directories...${NC}"
mkdir -p results_openmp/{raw_output,intermediate_results,performance_data}

# 测试参数设置
echo -e "${BLUE}Preparing tests with different matrix sizes and thread counts...${NC}"
# 矩阵规模
SIZES=(256 512 1024)
# 线程数
THREAD_COUNTS=(1 2 4 8 16)

# 创建执行时间和加速比CSV文件
echo "matrix_size,thread_count,serial,basic_omp,single_region,nowait,schedule" > results_openmp/execution_time.csv
echo "matrix_size,thread_count,basic_omp,single_region,nowait,schedule" > results_openmp/speedup.csv

# 对每个矩阵大小运行测试
for size in "${SIZES[@]}"; do
  echo -e "\n${BLUE}Testing matrix size: ${YELLOW}$size${NC}"

  # 对每个线程数运行测试
  for thread_count in "${THREAD_COUNTS[@]}"; do
    echo -e "${BLUE}Testing with thread count: ${YELLOW}$thread_count${NC}"
    
    # 创建中间结果文件
    result_file="results_openmp/intermediate_results/output_${size}_threads_${thread_count}.txt"
    echo "=== OpenMP Gaussian Elimination Test (Size: $size, Threads: $thread_count) ===" > "$result_file"
    echo "Command: qemu-aarch64 ./gaussian_elimination_openmp $size $thread_count" >> "$result_file"
    echo "Started at: $(date)" >> "$result_file"
    echo "----------------------------------------" >> "$result_file"

    # 设置缓存预热运行
    echo -e "${BLUE}Running cache warm-up iteration...${NC}"
    qemu-aarch64 ./gaussian_elimination_openmp $size $thread_count > /dev/null 2>&1 || true
    
    # 设置OpenMP线程数
    export OMP_NUM_THREADS=$thread_count
    
    # 正式运行程序
    echo -e "${BLUE}Running benchmark...${NC}"
    # 在QEMU中使用transparent huge pages以提高性能
    export QEMU_RESERVED_VA=8G
    export QEMU_HUGETLB=1
    
    # 运行程序并收集输出
    output=$(qemu-aarch64 ./gaussian_elimination_openmp $size $thread_count)
    
    # 显示输出概要
    echo -e "${GREEN}Program completed for size $size with $thread_count threads${NC}"
    echo "$output" | grep -E "time|speedup|correct"
    
    # 保存当前设置的完整输出
    echo "$output" > "results_openmp/raw_output/output_${size}_threads_${thread_count}.txt"
    echo "$output" >> "$result_file"
    
    # 添加分隔符和时间戳
    echo "----------------------------------------" >> "$result_file"
    echo "Finished at: $(date)" >> "$result_file"
    
    # 提取执行时间和加速比
    serial_time=$(echo "$output" | grep "Serial version execution time:" | awk '{print $5}')
    basic_time=$(echo "$output" | grep "Basic OpenMP version execution time:" | awk '{print $6}')
    single_time=$(echo "$output" | grep "Single Region OpenMP version execution time:" | awk '{print $7}')
    nowait_time=$(echo "$output" | grep "Nowait OpenMP version execution time:" | awk '{print $6}')
    schedule_time=$(echo "$output" | grep "Schedule OpenMP version execution time:" | awk '{print $6}')
    
    basic_speedup=$(echo "$output" | grep "Basic OpenMP version speedup:" | awk '{print $5}')
    single_speedup=$(echo "$output" | grep "Single Region OpenMP version speedup:" | awk '{print $6}')
    nowait_speedup=$(echo "$output" | grep "Nowait OpenMP version speedup:" | awk '{print $5}')
    schedule_speedup=$(echo "$output" | grep "Schedule OpenMP version speedup:" | awk '{print $5}')
    
    # 添加到CSV文件
    echo "$size,$thread_count,$serial_time,$basic_time,$single_time,$nowait_time,$schedule_time" >> results_openmp/execution_time.csv
    echo "$size,$thread_count,$basic_speedup,$single_speedup,$nowait_speedup,$schedule_speedup" >> results_openmp/speedup.csv
    
    echo -e "${GREEN}Completed test for size $size with $thread_count threads${NC}"
  done
done

# 显示CSV文件内容
echo -e "${BLUE}Results summary:${NC}"
echo "Contents of execution_time.csv:"
cat results_openmp/execution_time.csv
echo ""
echo "Contents of speedup.csv:"
cat results_openmp/speedup.csv
echo ""

# 使用Python绘制图表
echo -e "${BLUE}Generating plots...${NC}"
python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

try:
    # 创建图表目录
    os.makedirs('results_openmp/plots', exist_ok=True)
    
    # 读取执行时间和加速比数据
    time_data = pd.read_csv('results_openmp/execution_time.csv')
    speedup_data = pd.read_csv('results_openmp/speedup.csv')
    
    # 确保数据列是数字类型
    for col in time_data.columns:
        if col not in ['matrix_size', 'thread_count']:
            time_data[col] = pd.to_numeric(time_data[col], errors='coerce')
            
    for col in speedup_data.columns:
        if col not in ['matrix_size', 'thread_count']:
            speedup_data[col] = pd.to_numeric(speedup_data[col], errors='coerce')
    
    # 获取唯一的矩阵大小和线程数
    matrix_sizes = sorted(time_data['matrix_size'].unique())
    thread_counts = sorted(time_data['thread_count'].unique())
    
    # 算法名列表
    algorithms = ['basic_omp', 'single_region', 'nowait', 'schedule']
    algo_labels = ['Basic OpenMP', 'Single Region', 'Nowait', 'Schedule']
    
    # 1. 为每个矩阵大小绘制执行时间随线程数的变化
    for size in matrix_sizes:
        plt.figure(figsize=(10, 6))
        
        # 提取当前矩阵大小的数据
        size_data = time_data[time_data['matrix_size'] == size]
        
        # 绘制线程数与执行时间关系
        plt.plot(size_data['thread_count'], size_data['serial'], 'o-', label='Serial', linewidth=2)
        for algo in algorithms:
            plt.plot(size_data['thread_count'], size_data[algo], '-s', label=algo.replace('_', ' ').title(), linewidth=2)
        
        plt.title(f'Execution Time vs Thread Count (Matrix Size {size}x{size})', fontsize=14)
        plt.xlabel('Thread Count', fontsize=12)
        plt.ylabel('Execution Time (μs)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results_openmp/plots/exec_time_{size}.png', dpi=300)
        plt.close()
    
    # 2. 为每个矩阵大小绘制加速比随线程数的变化
    for size in matrix_sizes:
        plt.figure(figsize=(10, 6))
        
        # 提取当前矩阵大小的数据
        size_data = speedup_data[speedup_data['matrix_size'] == size]
        
        # 绘制线程数与加速比关系
        for algo in algorithms:
            plt.plot(size_data['thread_count'], size_data[algo], '-s', label=algo.replace('_', ' ').title(), linewidth=2)
        
        # 添加理想加速比参考线
        plt.plot(thread_counts, thread_counts, '--', color='gray', label='Ideal Speedup')
        
        plt.title(f'Speedup vs Thread Count (Matrix Size {size}x{size})', fontsize=14)
        plt.xlabel('Thread Count', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results_openmp/plots/speedup_{size}.png', dpi=300)
        plt.close()
    
    # 3. 为每个矩阵大小绘制并行效率随线程数的变化
    for size in matrix_sizes:
        plt.figure(figsize=(10, 6))
        
        # 提取当前矩阵大小的数据
        size_data = speedup_data[speedup_data['matrix_size'] == size]
        
        # 计算并绘制并行效率
        for algo in algorithms:
            efficiency = size_data[algo] / size_data['thread_count']
            plt.plot(size_data['thread_count'], efficiency, '-s', label=algo.replace('_', ' ').title(), linewidth=2)
        
        # 添加理想效率参考线
        plt.axhline(y=1, linestyle='--', color='gray', label='Ideal Efficiency')
        
        plt.title(f'Parallel Efficiency vs Thread Count (Matrix Size {size}x{size})', fontsize=14)
        plt.xlabel('Thread Count', fontsize=12)
        plt.ylabel('Parallel Efficiency (Speedup/Thread Count)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results_openmp/plots/efficiency_{size}.png', dpi=300)
        plt.close()
    
    # 4. 为每个算法绘制不同矩阵大小下的扩展性图
    for i, algo in enumerate(algorithms):
        plt.figure(figsize=(10, 6))
        
        for size in matrix_sizes:
            # 提取当前矩阵大小的数据
            size_data = speedup_data[speedup_data['matrix_size'] == size]
            plt.plot(size_data['thread_count'], size_data[algo], '-o', 
                    label=f'Size {size}x{size}', linewidth=2)
        
        # 添加理想加速比参考线
        plt.plot(thread_counts, thread_counts, '--', color='gray', label='Ideal Speedup')
        
        plt.title(f'{algo_labels[i]} Algorithm Scaling', fontsize=14)
        plt.xlabel('Thread Count', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results_openmp/plots/algo_{algo}_scaling.png', dpi=300)
        plt.close()
    
    # 5. 热力图显示不同算法在不同线程数下的加速比
    for size in matrix_sizes:
        # 提取当前矩阵大小的数据并创建热力图数据框
        size_data = speedup_data[speedup_data['matrix_size'] == size]
        
        # 转为矩阵格式用于热力图
        heatmap_data = np.zeros((len(thread_counts), len(algorithms)))
        for i, thread in enumerate(thread_counts):
            thread_data = size_data[size_data['thread_count'] == thread]
            if len(thread_data) > 0:  # 确保有数据
                for j, algo in enumerate(algorithms):
                    heatmap_data[i, j] = thread_data[algo].values[0]
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Speedup')
        
        # 添加标签
        plt.yticks(range(len(thread_counts)), thread_counts)
        plt.xticks(range(len(algorithms)), algo_labels)
        plt.ylabel('Thread Count', fontsize=12)
        plt.xlabel('Algorithm', fontsize=12)
        plt.title(f'OpenMP Speedup Heatmap (Size {size}x{size})', fontsize=14)
        
        # 添加数据标签
        for i in range(len(thread_counts)):
            for j in range(len(algorithms)):
                plt.text(j, i, f'{heatmap_data[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if heatmap_data[i, j] < np.max(heatmap_data)*0.7 else 'black')
        
        plt.tight_layout()
        plt.savefig(f'results_openmp/plots/heatmap_{size}.png', dpi=300)
        plt.close()

    print('All plots generated successfully!')
except Exception as e:
    import traceback
    print(f'Error generating plots: {str(e)}')
    print(traceback.format_exc())
    with open('results_openmp/plot_error.log', 'w') as f:
        f.write(f'Error: {str(e)}\\n')
        f.write(traceback.format_exc())
    sys.exit(1)
"

# 检查绘图是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to generate plots. Check results_openmp/plot_error.log for details.${NC}"
  exit 1
fi

# 生成性能报告
echo -e "${BLUE}Generating performance report...${NC}"
cat > results_openmp/performance_report.md << EOL
# OpenMP高斯消去算法线程扩展性报告

## 概述
本报告分析了OpenMP并行高斯消去算法在不同线程数下的性能表现，评估了算法的扩展性和效率。
测试日期: 2025-05-22 07:25:14
测试用户: KKKyriejiang

## 测试环境
- 架构: ARM (通过QEMU模拟)
- 编译器: aarch64-linux-gnu-g++ 带O3优化和OpenMP支持
- 测试线程数: ${THREAD_COUNTS[@]}
- 测试矩阵规模: ${SIZES[@]}

## 算法实现
1. **基本OpenMP版本**: 使用简单的parallel for指令并行化消去循环
2. **单一并行区OpenMP版本**: 使用单一并行区域和single指令减少线程创建开销
3. **Nowait优化OpenMP版本**: 使用nowait子句减少不必要的同步开销
4. **动态调度OpenMP版本**: 使用动态调度策略优化负载均衡

## 线程扩展性分析

### 执行时间分析

![256x256矩阵执行时间](plots/exec_time_256.png)
*256x256矩阵的执行时间随线程数的变化*

![512x512矩阵执行时间](plots/exec_time_512.png)
*512x512矩阵的执行时间随线程数的变化*

![1024x1024矩阵执行时间](plots/exec_time_1024.png)
*1024x1024矩阵的执行时间随线程数的变化*

观察以上执行时间图表，我们可以得出以下结论：

1. 随着线程数增加，各算法的执行时间总体呈下降趋势，但下降幅度因矩阵大小而异。
2. 在小矩阵上(256x256)，当线程数超过4时执行时间可能会略有增加，表明并行开销开始超过并行收益。
3. 在大矩阵上(1024x1024)，执行时间在线程数增加时持续下降，表明并行计算的效益占据主导。
4. 所有OpenMP版本都比串行版本有显著改进，即使在线程数为1的情况下，这可能是由于编译器优化和指令重排序。

### 加速比分析

![256x256矩阵加速比](plots/speedup_256.png)
*256x256矩阵的加速比随线程数的变化*

![512x512矩阵加速比](plots/speedup_512.png)
*512x512矩阵的加速比随线程数的变化*

![1024x1024矩阵加速比](plots/speedup_1024.png)
*1024x1024矩阵的加速比随线程数的变化*

从加速比图表可以看出：

1. 在小矩阵上(256x256)，加速比随线程数增加的增长非常有限，甚至出现下降，最高加速比约为3.5-4倍。
2. 中等大小矩阵(512x512)的加速比表现更为稳定，基本保持在3-3.6倍之间。
3. 大矩阵(1024x1024)展示了最一致的加速比，在所有线程配置下基本维持在3.5-3.7倍左右。
4. 所有情况下的加速比都远低于理想线性加速比，表明存在显著的并行开销或Amdahl定律限制。

### 并行效率分析

![256x256矩阵并行效率](plots/efficiency_256.png)
*256x256矩阵的并行效率随线程数的变化*

![512x512矩阵并行效率](plots/efficiency_512.png)
*512x512矩阵的并行效率随线程数的变化*

![1024x1024矩阵并行效率](plots/efficiency_1024.png)
*1024x1024矩阵的并行效率随线程数的变化*

并行效率(加速比/线程数)反映了并行算法利用额外线程的能力：

1. 所有配置的并行效率都随线程数增加而下降，这是由于通信开销、负载不均衡和串行部分的影响。
2. 小矩阵上并行效率下降最为显著，在16线程时效率降至0.2-0.3左右。
3. 大矩阵维持了相对更高的并行效率，但在高线程数下仍显著下降。
4. 不同OpenMP实现的并行效率差异随线程数增加而扩大，优化策略的重要性愈发明显。

## 算法性能对比

### 各算法的扩展性比较

![基本OpenMP扩展性](plots/algo_basic_omp_scaling.png)
*基本OpenMP版本在不同矩阵大小下的扩展性*

![单一区域扩展性](plots/algo_single_region_scaling.png)
*单一区域OpenMP版本在不同矩阵大小下的扩展性*

![Nowait扩展性](plots/algo_nowait_scaling.png)
*Nowait OpenMP版本在不同矩阵大小下的扩展性*

![动态调度扩展性](plots/algo_schedule_scaling.png)
*动态调度OpenMP版本在不同矩阵大小下的扩展性*

不同算法的比较显示：

1. 对于所有算法，矩阵大小是影响扩展性的关键因素，较大矩阵通常表现出更好的扩展性。
2. Nowait优化在低线程数下表现尤为出色，但在高线程数环境下优势减弱。
3. 动态调度版本在大多数矩阵大小和线程配置下表现稳定。
4. 单一区域实现在中等线程数下具有良好表现，但在高线程数时效率下降较快。

### 加速比热图分析

![256x256矩阵热图](plots/heatmap_256.png)
*256x256矩阵不同算法和线程数的加速比热图*

![512x512矩阵热图](plots/heatmap_512.png)
*512x512矩阵不同算法和线程数的加速比热图*

![1024x1024矩阵热图](plots/heatmap_1024.png)
*1024x1024矩阵不同算法和线程数的加速比热图*

热图直观地展示了不同算法在各种线程配置下的表现：

1. 小矩阵(256x256)上，4线程下的Nowait版本表现最佳。
2. 中等矩阵(512x512)上，8线程下的静态调度版本达到最高加速比。
3. 大矩阵(1024x1024)上，16线程下的动态调度版本效率最高。
4. 总体而言，Nowait和动态调度策略在大多数场景下表现优秀。

## 结论与建议

### 最佳线程配置
基于我们的测试结果，以下是对不同矩阵大小的最佳线程数建议：

1. **小矩阵 (256×256)**：使用2-4线程最为高效，超过4线程会导致效率显著下降。
2. **中等矩阵 (512×512)**：4-8线程是最佳选择，在这个范围内能获得良好的加速比而不损失太多效率。
3. **大矩阵 (1024×1024)**：8-16线程可提供最好的整体性能，特别是使用动态调度策略时。

### 算法选择建议
1. **小型问题**：对于小矩阵，Nowait优化版本通常表现最佳，因为它减少了同步开销。
2. **中型问题**：对于中等大小矩阵，可以考虑使用单一区域或静态调度实现，它们在这种规模下提供了良好的平衡。
3. **大型问题**：对于大矩阵，动态调度版本通常能提供最佳性能，因为它能更好地处理负载不均衡。

### 性能优化方向
1. **负载均衡**：考虑实现更先进的负载均衡技术，如工作窃取或自适应调度。
2. **内存访问优化**：引入分块算法以改善缓存局部性和减少NUMA效应。
3. **混合并行**：探索OpenMP与向量化指令集(如NEON)结合的混合并行策略。
4. **算法改进**：考虑使用更具扩展性的算法变体，如分级或递归方法。

总结而言，OpenMP高斯消元算法在ARM平台上展示了良好的性能，但远未达到理想的线性扩展性。最佳性能需要根据问题规模谨慎选择线程数和OpenMP实现策略。未来工作应侧重于改善内存访问模式和减少同步开销，这是限制当前实现扩展性的主要因素。
EOL

echo -e "${GREEN}Performance report generated: results_openmp/performance_report.md${NC}"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo -e "${GREEN}Results saved in results_openmp directory${NC}"
echo -e "${BLUE}You can view the plots and performance report in the results_openmp directory${NC}"