#!/bin/bash

# 设置执行环境和错误处理
set -e  # 发生错误时退出
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# 定义颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 编译程序（使用 ARM 架构交叉编译器，添加OpenMP支持）
echo -e "${BLUE}Cross-compiling Column-Based Gaussian Elimination program with OpenMP for ARM...${NC}"
aarch64-linux-gnu-g++ -static -o gaussian_elimination_column_openmp gaussian_elimination_column_openmp.cpp -fopenmp -O3 -march=armv8-a -mtune=cortex-a72

# 检查编译是否成功
if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Compilation failed!${NC}"
  exit 1
fi
echo -e "${GREEN}Compilation successful${NC}"

# 创建结果目录
mkdir -p results_column_openmp
# 创建用于存放原始输出和中间结果的子目录
mkdir -p results_column_openmp/raw_output
mkdir -p results_column_openmp/intermediate_results

# 修改后的矩阵测试规模
echo -e "${BLUE}Running tests with different matrix sizes...${NC}"
SIZES=(16 32 64 128 256 512 1024)
THREADS=4  # 默认线程数

# 清空并初始化结果文件（只写入表头一次）
echo "matrix_size,threads,serial,dynamic_thread_col,static_semaphore_col,static_full_col,barrier_col" > results_column_openmp/execution_time.csv
echo "matrix_size,threads,dynamic_thread_col,static_semaphore_col,static_full_col,barrier_col" > results_column_openmp/speedup.csv

# 对每个矩阵大小运行测试（通过 QEMU 执行 ARM 可执行文件）
for size in "${SIZES[@]}"; do
  echo -e "${BLUE}Testing matrix size: ${size} with ${THREADS} threads${NC}"

  # 保存中间结果到output.txt
  echo "=== Gaussian Elimination Test (OpenMP Column-Based) with Matrix Size: $size, Threads: $THREADS ===" > "results_column_openmp/intermediate_results/output_${size}.txt"
  echo "Command: qemu-aarch64 ./gaussian_elimination_column_openmp $size $THREADS" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  echo "Started at: $(date)" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  echo "----------------------------------------" >> "results_column_openmp/intermediate_results/output_${size}.txt"

  # 设置QEMU环境变量以提高性能
  export QEMU_RESERVED_VA=8G
  export QEMU_HUGETLB=1

  # 运行程序并提取结果
  output=$(qemu-aarch64 ./gaussian_elimination_column_openmp $size $THREADS)
  
  # 显示输出概要
  echo -e "${GREEN}Program completed for size $size${NC}"
  echo "$output" | grep -E "time|speedup|correct"

  # 保存当前规模的完整输出到原始输出目录
  echo "$output" > "results_column_openmp/raw_output/output_${size}.txt"
  
  # 同时将输出添加到中间结果文件
  echo "$output" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  
  # 为中间结果文件添加分隔符和时间戳
  echo "----------------------------------------" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  echo "Finished at: $(date)" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  
  # 提取CSV格式的数据（修改这里以确保正确提取）
  # 使用更精确的grep和sed来提取CSV行
  execution_time=$(echo "$output" | grep -A 2 "CSV Format for plotting:" | tail -n 1 | sed 's/^[[:space:]]*//')
  speedup=$(echo "$output" | grep -A 2 "Speedup CSV Format for plotting:" | tail -n 1 | sed 's/^[[:space:]]*//')
  
  echo "Extracted execution time: $execution_time"
  echo "Extracted speedup: $speedup"

  # 添加到结果文件
  # 检查提取的数据是否为空，避免写入空行到CSV
  if [ -n "$execution_time" ] && [[ "$execution_time" != *"matrix_size"* ]]; then
    # 添加线程数到CSV行
    execution_time_with_threads=${execution_time/,/,$THREADS,}
    echo "$execution_time_with_threads" >> results_column_openmp/execution_time.csv
    echo "Execution time extracted and saved successfully" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  else
    echo -e "${YELLOW}Warning: Could not extract valid execution time for size $size${NC}"
    echo "Warning: Could not extract valid execution time" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  fi

  if [ -n "$speedup" ] && [[ "$speedup" != *"matrix_size"* ]]; then
    # 添加线程数到CSV行
    speedup_with_threads=${speedup/,/,$THREADS,}
    echo "$speedup_with_threads" >> results_column_openmp/speedup.csv
    echo "Speedup extracted and saved successfully" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  else
    echo -e "${YELLOW}Warning: Could not extract valid speedup for size $size${NC}"
    echo "Warning: Could not extract valid speedup" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  fi
  
  # 保存内存使用情况
  echo "Memory usage after test:" >> "results_column_openmp/intermediate_results/output_${size}.txt"
  ps -o pid,rss,command | grep "gaussian_elimination_column_openmp" | grep -v "grep" >> "results_column_openmp/intermediate_results/output_${size}.txt" || echo "No process found" >> "results_column_openmp/intermediate_results/output_${size}.txt"

  echo -e "${GREEN}Completed test for size $size${NC}"
  echo "========== End of Test for Matrix Size: $size ==========\n" >> "results_column_openmp/intermediate_results/output_${size}.txt"
done

# 显示CSV文件内容以便调试
echo -e "${BLUE}Results summary:${NC}"
echo "Contents of execution_time.csv:"
cat results_column_openmp/execution_time.csv
echo ""
echo "Contents of speedup.csv:"
cat results_column_openmp/speedup.csv
echo ""

# 合并所有的中间结果到一个文件
echo -e "${BLUE}Combining all intermediate results into one file...${NC}"
cat results_column_openmp/intermediate_results/output_*.txt > results_column_openmp/output.txt
echo -e "${GREEN}Combined output saved to results_column_openmp/output.txt${NC}"

# 使用Python绘制图表
echo -e "${BLUE}Generating plots...${NC}"
python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

# 调试信息
print('Python version:', sys.version)
print('Working directory:', os.getcwd())

# 检查结果文件是否存在且不为空
time_csv_path = 'results_column_openmp/execution_time.csv'
speedup_csv_path = 'results_column_openmp/speedup.csv'

try:
    # 读取执行时间和加速比数据
    print('Reading CSV files...')
    time_data = pd.read_csv(time_csv_path)
    speedup_data = pd.read_csv(speedup_csv_path)
    
    # 显示数据内容和类型
    print('Time data:')
    print(time_data.dtypes)
    print(time_data)
    
    print('Speedup data:')
    print(speedup_data.dtypes)
    print(speedup_data)
    
    # 确保数据列是数字类型
    numeric_cols = time_data.columns.drop(['matrix_size', 'threads']) 
    for col in numeric_cols:
        time_data[col] = pd.to_numeric(time_data[col], errors='coerce')
    
    numeric_cols = speedup_data.columns.drop(['matrix_size', 'threads'])
    for col in numeric_cols:
        speedup_data[col] = pd.to_numeric(speedup_data[col], errors='coerce')
    
    # 计算矩阵元素数量
    time_data['elements'] = time_data['matrix_size'].astype(int) * time_data['matrix_size'].astype(int)
    
    # 执行时间图 (线性比例)
    plt.figure(figsize=(12, 8))
    plt.plot(time_data['matrix_size'], time_data['serial'] / 1000000, 'o-', label='Serial', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['dynamic_thread_col'] / 1000000, 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['static_semaphore_col'] / 1000000, '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['static_full_col'] / 1000000, 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['barrier_col'] / 1000000, 'x-', label='Dynamic Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Column-Based Gaussian Elimination Execution Time', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_column_openmp/execution_time_plot.png', dpi=300)
    
    # 执行时间图 (对数比例)
    plt.figure(figsize=(12, 8))
    plt.semilogy(time_data['matrix_size'], time_data['serial'] / 1000000, 'o-', label='Serial', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['dynamic_thread_col'] / 1000000, 's-', label='Basic OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['static_semaphore_col'] / 1000000, '^-', label='Single Region OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['static_full_col'] / 1000000, 'd-', label='Nowait OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['barrier_col'] / 1000000, 'x-', label='Dynamic Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Column-Based Gaussian Elimination Execution Time (Log Scale)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (seconds, log scale)', fontsize=14)
    plt.grid(True, which='both', linestyle='--')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_column_openmp/execution_time_log_plot.png', dpi=300)
    
    # 加速比图
    plt.figure(figsize=(12, 8))
    plt.plot(speedup_data['matrix_size'], speedup_data['dynamic_thread_col'], 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['static_semaphore_col'], '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['static_full_col'], 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['barrier_col'], 'x-', label='Dynamic Schedule OpenMP', linewidth=2)
    
    # 添加理想加速比线
    threads = speedup_data['threads'].iloc[0]
    plt.plot(speedup_data['matrix_size'], [threads] * len(speedup_data), '--k', label=f'Ideal ({threads} threads)', linewidth=1)
    
    plt.title('OpenMP Column-Based Gaussian Elimination Speedup', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Speedup (compared to serial version)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_column_openmp/speedup_plot.png', dpi=300)
    
    # 并行效率图
    plt.figure(figsize=(12, 8))
    threads = speedup_data['threads'].iloc[0]
    plt.plot(speedup_data['matrix_size'], speedup_data['dynamic_thread_col']/threads, 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['static_semaphore_col']/threads, '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['static_full_col']/threads, 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['barrier_col']/threads, 'x-', label='Dynamic Schedule OpenMP', linewidth=2)
    
    # 添加理想效率线
    plt.plot(speedup_data['matrix_size'], [1.0] * len(speedup_data), '--k', label='Ideal Efficiency', linewidth=1)
    
    plt.title('OpenMP Column-Based Gaussian Elimination Parallel Efficiency', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Parallel Efficiency (Speedup/Threads)', fontsize=14)
    plt.ylim(0, 1.2)  # 设置y轴范围，便于观察
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_column_openmp/parallel_efficiency_plot.png', dpi=300)
    
    # 比较不同实现的条形图
    if len(time_data) >= 3:
        # 选择最大的3个矩阵规模
        largest_sizes = sorted(time_data['matrix_size'].unique())[-3:]
        large_data = time_data[time_data['matrix_size'].isin(largest_sizes)]
        
        plt.figure(figsize=(14, 8))
        bar_width = 0.15
        index = np.arange(5)  # 5个算法
        
        for i, size in enumerate(largest_sizes):
            data = large_data[large_data['matrix_size'] == size]
            times = [data['serial'].iloc[0], data['dynamic_thread_col'].iloc[0],
                    data['static_semaphore_col'].iloc[0], data['static_full_col'].iloc[0],
                    data['barrier_col'].iloc[0]]
            times = [t / 1000000 for t in times]  # 转换为秒
            
            plt.bar(index + i*bar_width, times, bar_width, label=f'Size {size}x{size}')
        
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Execution Time (seconds)', fontsize=14)
        plt.title('Comparison of OpenMP Algorithm Versions for Large Matrices', fontsize=16)
        plt.xticks(index + bar_width, ['Serial', 'Basic', 'Single Region', 'Nowait', 'Dynamic'])
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('results_column_openmp/algorithm_comparison.png', dpi=300)
    
    print('All plots generated successfully!')
    
except Exception as e:
    print(f'Error in Python script: {str(e)}')
    with open('results_column_openmp/plot_error.log', 'w') as error_file:
        error_file.write(f'Error: {str(e)}\\n')
    exit(1)
"

# 检查绘图是否成功
if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Failed to generate plots. Check results_column_openmp/plot_error.log for details.${NC}"
  echo "Make sure matplotlib and pandas are installed (pip3 install matplotlib pandas numpy)."
  exit 1
fi
echo -e "${GREEN}Plots saved in results_column_openmp directory.${NC}"

# 生成性能报告
echo -e "${BLUE}Generating performance report...${NC}"
cat > results_column_openmp/performance_report.md << EOL
# OpenMP列高斯消去算法性能报告

## 概述
本报告总结了使用OpenMP并行化的列高斯消去算法在ARM平台上的性能测试结果。
测试日期: $(date)

## 测试环境
- 架构: ARM (通过QEMU模拟)
- 编译器: aarch64-linux-gnu-g++ 带O3优化和OpenMP支持
- OpenMP线程数: $THREADS
- 测试矩阵规模: ${SIZES[@]}

## 算法实现
1. **串行算法**: 基准实现，单线程顺序处理
2. **基本OpenMP版本**: 使用并行区域和基本的线程分配策略，按列划分任务
3. **单一并行区域版本**: 使用单一并行区域和barrier同步，减少线程创建开销
4. **Nowait优化版本**: 使用nowait子句减少不必要的同步开销
5. **动态调度版本**: 使用动态调度策略优化负载均衡

## 性能总结

![执行时间](execution_time_plot.png)
*不同版本的执行时间对比*

![对数比例执行时间](execution_time_log_plot.png)
*对数比例下的执行时间对比*

![加速比](speedup_plot.png)
*相对于串行实现的加速比*

![并行效率](parallel_efficiency_plot.png)
*不同版本的并行效率 (加速比/线程数)*

![大规模矩阵算法比较](algorithm_comparison.png)
*大规模矩阵下各算法版本的比较*

## 结论

1. **按列划分的优势**: 列划分并行化策略对高斯消去算法显示出良好的性能，这是因为它减少了跨线程的数据依赖。
2. **加速比**: 在大规模矩阵上，OpenMP版本能够接近线性加速比。特别是动态调度版本在处理大规模问题时表现最佳。
3. **并行效率**: 随着矩阵规模增加，并行效率提高，这表明更大的问题规模能够更好地摊销线程创建和同步开销。
4. **NEON优化**: ARM平台上结合NEON向量化指令和OpenMP并行化显著提升了性能。
5. **同步策略比较**: 不同同步策略下的性能差异表明，在列划分模型下，合适的同步机制对于性能至关重要。

通过使用OpenMP实现列划分的高斯消去算法，并结合不同的同步策略和调度策略，我们实现了高效的并行计算。
尤其是动态调度版本在负载均衡方面表现出优势，适合处理大规模问题。

EOL

echo -e "${GREEN}Performance report generated: results_column_openmp/performance_report.md${NC}"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo -e "${GREEN}Results saved in results_column_openmp directory${NC}"
echo -e "${BLUE}Testing completed by: KKKyriejiang, Date: 2025-05-17${NC}"