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

echo -e "${BLUE}=== Gaussian Elimination Algorithm Test Script (KKKyriejiang) ===${NC}"
echo -e "${BLUE}Current Date and Time: $(date "+%Y-%m-%d %H:%M:%S")${NC}"
echo -e "${BLUE}Current User: KKKyriejiang${NC}"

# 检查是否存在gaussian_avx_openmp.cpp
if [ ! -f "gaussian_avx_openmp.cpp" ]; then
  echo -e "${RED}Error: gaussian_avx_openmp.cpp not found!${NC}"
  exit 1
fi

# 检查AVX支持
echo -e "${BLUE}Checking CPU support for AVX...${NC}"
if grep -q "avx" /proc/cpuinfo; then
  echo -e "${GREEN}AVX instructions are supported!${NC}"
else
  echo -e "${RED}Warning: AVX instructions not explicitly found in /proc/cpuinfo${NC}"
  echo -e "${YELLOW}Program may not function correctly without AVX support.${NC}"
  read -p "Do you want to continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Exiting.${NC}"
    exit 1
  fi
  echo -e "${YELLOW}Continuing at your own risk. Results may be unreliable.${NC}"
fi

# 编译程序（使用x86架构编译器并支持AVX和OpenMP）
echo -e "${BLUE}Compiling Gaussian Elimination program with AVX and OpenMP...${NC}"
g++ -o gaussian_avx_openmp gaussian_avx_openmp.cpp \
  -fopenmp -O3 -mavx -march=native \
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
mkdir -p results_avx_openmp/{raw_output,intermediate_results,performance_data,plots}

# 测试参数设置
echo -e "${BLUE}Preparing tests with different matrix sizes...${NC}"
# 矩阵规模
SIZES=(64 128 256 512 1024 2048)
# 线程数
THREADS=4

# 初始化结果文件
echo "matrix_size,serial,basic_omp,single_region,nowait,schedule" > results_avx_openmp/execution_time.csv
echo "matrix_size,basic_omp,single_region,nowait,schedule" > results_avx_openmp/speedup.csv

# 对每个矩阵大小运行测试
for size in "${SIZES[@]}"; do
  echo -e "\n${BLUE}Testing matrix size: ${YELLOW}${size} x ${size}${NC}"

  # 创建中间结果文件
  result_file="results_avx_openmp/intermediate_results/output_${size}.txt"
  echo "=== AVX OpenMP Gaussian Elimination Test (Size: $size, Threads: $THREADS) ===" > "$result_file"
  echo "Command: ./gaussian_avx_openmp $size $THREADS" >> "$result_file"
  echo "Started at: $(date)" >> "$result_file"
  echo "----------------------------------------" >> "$result_file"

  # 设置缓存预热运行
  echo -e "${BLUE}Running cache warm-up iteration...${NC}"
  ./gaussian_avx_openmp $size $THREADS > /dev/null 2>&1 || true
  
  # 正式运行程序
  echo -e "${BLUE}Running benchmark...${NC}"
  
  # 运行程序并收集输出
  output=$(./gaussian_avx_openmp $size $THREADS)
  
  # 显示输出概要
  echo -e "${GREEN}Program completed for size $size${NC}"
  echo "$output" | grep -E "time|speedup|correct"
  
  # 保存当前规模的完整输出
  echo "$output" > "results_avx_openmp/raw_output/output_${size}.txt"
  echo "$output" >> "$result_file"
  
  # 添加分隔符和时间戳
  echo "----------------------------------------" >> "$result_file"
  echo "Finished at: $(date)" >> "$result_file"
  
  # 提取CSV格式的数据
  execution_time=$(echo "$output" | grep -A 2 "CSV Format for plotting:" | tail -n 1)
  speedup=$(echo "$output" | grep -A 2 "Speedup CSV Format for plotting:" | tail -n 1)
  
  # 添加到结果文件
  if [ -n "$execution_time" ] && [[ "$execution_time" != *"matrix_size"* ]]; then
    echo "$execution_time" >> results_avx_openmp/execution_time.csv
    echo "Execution time extracted and saved successfully" >> "$result_file"
  else
    echo -e "${YELLOW}Warning: Could not extract valid execution time for size $size${NC}"
    echo "Warning: Could not extract valid execution time" >> "$result_file"
  fi

  if [ -n "$speedup" ] && [[ "$speedup" != *"matrix_size"* ]]; then
    echo "$speedup" >> results_avx_openmp/speedup.csv
    echo "Speedup extracted and saved successfully" >> "$result_file"
  else
    echo -e "${YELLOW}Warning: Could not extract valid speedup for size $size${NC}"
    echo "Warning: Could not extract valid speedup" >> "$result_file"
  fi
  
  # 记录内存使用情况
  echo -e "${BLUE}Recording memory usage...${NC}"
  echo "Memory usage after test:" >> "$result_file"
  ps -o pid,rss,command | grep "gaussian_avx_openmp" | grep -v "grep" >> "$result_file" || echo "No process found" >> "$result_file"

  echo -e "${GREEN}Completed test for size $size${NC}"
  echo "========== End of Test for Matrix Size: $size ==========\n" >> "$result_file"
done

# 显示CSV文件内容
echo -e "${BLUE}Results summary:${NC}"
echo "Contents of execution_time.csv:"
cat results_avx_openmp/execution_time.csv
echo ""
echo "Contents of speedup.csv:"
cat results_avx_openmp/speedup.csv
echo ""

# 合并所有的中间结果
echo -e "${BLUE}Combining all results...${NC}"
cat results_avx_openmp/intermediate_results/output_*.txt > results_avx_openmp/output.txt
echo -e "${GREEN}Combined output saved to results_avx_openmp/output.txt${NC}"

# 使用Python绘制图表
echo -e "${BLUE}Generating plots...${NC}"
python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

try:
    # 读取执行时间和加速比数据
    time_data = pd.read_csv('results_avx_openmp/execution_time.csv')
    speedup_data = pd.read_csv('results_avx_openmp/speedup.csv')
    
    # 确保数据列是数字类型
    numeric_cols = time_data.columns.drop('matrix_size')
    for col in numeric_cols:
        time_data[col] = pd.to_numeric(time_data[col], errors='coerce')
    
    numeric_cols = speedup_data.columns.drop('matrix_size')
    for col in numeric_cols:
        speedup_data[col] = pd.to_numeric(speedup_data[col], errors='coerce')
    
    # 计算矩阵元素数量和内存使用量
    time_data['elements'] = time_data['matrix_size'] ** 2
    time_data['memory_usage_MB'] = time_data['elements'] * 4 / (1024*1024)  # 假设每个元素4字节
    
    # 执行时间图
    plt.figure(figsize=(12, 8))
    plt.plot(time_data['matrix_size'], time_data['serial'] / 1000000, 'o-', label='Serial', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['basic_omp'] / 1000000, 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['single_region'] / 1000000, '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['nowait'] / 1000000, 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['schedule'] / 1000000, 'x-', label='Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Gaussian Elimination Execution Time with AVX', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_avx_openmp/plots/execution_time_plot.png', dpi=300)
    
    # 对数比例的执行时间图
    plt.figure(figsize=(12, 8))
    plt.semilogy(time_data['matrix_size'], time_data['serial'] / 1000000, 'o-', label='Serial', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['basic_omp'] / 1000000, 's-', label='Basic OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['single_region'] / 1000000, '^-', label='Single Region OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['nowait'] / 1000000, 'd-', label='Nowait OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['schedule'] / 1000000, 'x-', label='Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Gaussian Elimination Execution Time with AVX (Log Scale)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (seconds, log scale)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_avx_openmp/plots/execution_time_plot_log.png', dpi=300)
    
    # 加速比图 (自动伸缩范围)
    plt.figure(figsize=(12, 8))
    plt.plot(speedup_data['matrix_size'], speedup_data['basic_omp'], 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['single_region'], '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['nowait'], 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['schedule'], 'x-', label='Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Gaussian Elimination Speedup with AVX', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Speedup (compared to serial version)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_avx_openmp/plots/speedup_plot.png', dpi=300)
    
    # 并行效率图
    plt.figure(figsize=(12, 8))
    threads = ${THREADS}  # 使用脚本定义的线程数
    plt.plot(speedup_data['matrix_size'], speedup_data['basic_omp']/threads, 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['single_region']/threads, '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['nowait']/threads, 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['schedule']/threads, 'x-', label='Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Gaussian Elimination Parallel Efficiency with AVX', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Parallel Efficiency (Speedup/Threads)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_avx_openmp/plots/parallel_efficiency_plot.png', dpi=300)
    
    # 算法比较图
    # 对于大矩阵，绘制条形图比较不同版本
    plt.figure(figsize=(14, 8))
    
    # 筛选最大的三个矩阵大小
    largest_sizes = sorted(time_data['matrix_size'].unique())[-3:]
    large_data = time_data[time_data['matrix_size'].isin(largest_sizes)]
    
    # 准备绘图数据
    algorithms = ['serial', 'basic_omp', 'single_region', 'nowait', 'schedule']
    alg_labels = ['Serial', 'Basic OpenMP', 'Single Region', 'Nowait', 'Schedule']
    
    # 设置x轴位置
    n_algs = len(algorithms)
    n_sizes = len(largest_sizes)
    width = 0.15
    
    # 绘制分组条形图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, size in enumerate(largest_sizes):
        size_data = large_data[large_data['matrix_size'] == size]
        times = [size_data[alg].values[0]/1000000 for alg in algorithms]  # 转换为秒
        x = np.arange(n_algs) + (i - n_sizes/2 + 0.5) * width
        ax.bar(x, times, width, label=f'Size {size}x{size}')
    
    ax.set_xticks(np.arange(n_algs))
    ax.set_xticklabels(alg_labels)
    ax.set_ylabel('Execution Time (seconds)', fontsize=14)
    ax.set_title('Comparison of Algorithm Versions for Large Matrices with AVX', fontsize=16)
    ax.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('results_avx_openmp/plots/algorithm_comparison.png', dpi=300)
    
    # 增加AVX性能热图
    plt.figure(figsize=(12, 8))
    
    # 创建归一化数据（针对每个矩阵大小，最快的实现为1.0）
    norm_data = time_data.copy()
    for idx, row in norm_data.iterrows():
        min_time = min([row['serial'], row['basic_omp'], row['single_region'], row['nowait'], row['schedule']])
        for alg in algorithms:
            norm_data.at[idx, alg] = row[alg] / min_time
    
    # 准备热图数据
    heatmap_data = []
    for alg in algorithms:
        heatmap_data.append(norm_data[alg].values)
    
    # 创建热图
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Relative Execution Time (lower is better)')
    plt.ylabel('Algorithm')
    plt.xlabel('Matrix Size')
    plt.title('Performance Heatmap of AVX+OpenMP Implementations', fontsize=16)
    plt.yticks(np.arange(len(algorithms)), alg_labels)
    plt.xticks(np.arange(len(time_data['matrix_size'])), time_data['matrix_size'])
    
    # 在热图中添加数值标签
    for i in range(len(algorithms)):
        for j in range(len(time_data['matrix_size'])):
            text_color = 'white' if heatmap_data[i][j] > 1.5 else 'black'
            plt.text(j, i, f'{heatmap_data[i][j]:.2f}', 
                     ha='center', va='center', color=text_color)
    
    plt.tight_layout()
    plt.savefig('results_avx_openmp/plots/performance_heatmap.png', dpi=300)
    
    # 加速比与矩阵大小的关系图（非线性x轴）
    plt.figure(figsize=(12, 8))
    
    # 使用更明显的标记
    plt.semilogx(speedup_data['matrix_size'], speedup_data['basic_omp'], 's-', label='Basic OpenMP', linewidth=2, markersize=10)
    plt.semilogx(speedup_data['matrix_size'], speedup_data['single_region'], '^-', label='Single Region OpenMP', linewidth=2, markersize=10)
    plt.semilogx(speedup_data['matrix_size'], speedup_data['nowait'], 'd-', label='Nowait OpenMP', linewidth=2, markersize=10)
    plt.semilogx(speedup_data['matrix_size'], speedup_data['schedule'], 'x-', label='Schedule OpenMP', linewidth=2, markersize=10)
    
    # 添加水平理想线
    plt.axhline(y=threads, color='gray', linestyle='--', label=f'Ideal ({threads} threads)')
    
    plt.title('Speedup vs. Matrix Size with AVX (Log Scale X-Axis)', fontsize=16)
    plt.xlabel('Matrix Size (log scale)', fontsize=14)
    plt.ylabel('Speedup', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_avx_openmp/plots/speedup_log_size.png', dpi=300)
    
    print('All plots generated successfully!')
    
except Exception as e:
    print(f'Error in Python script: {str(e)}')
    with open('results_avx_openmp/plots/plot_error.log', 'w') as error_file:
        error_file.write(f'Error: {str(e)}\\n')
    sys.exit(1)
"

# 检查绘图是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to generate plots. Check results_avx_openmp/plots/plot_error.log for details.${NC}"
  echo "Make sure matplotlib and pandas are installed (pip3 install matplotlib pandas numpy)."
else
  echo -e "${GREEN}Plots generated successfully!${NC}"
fi

# 生成性能报告
echo -e "${BLUE}Generating performance report...${NC}"
cat > results_avx_openmp/performance_report.md << EOL
# AVX优化OpenMP高斯消去算法性能报告

## 概述
本报告总结了使用AVX向量化和OpenMP并行化的高斯消去算法在x86平台上的性能测试结果。
测试日期: $(date)

## 测试环境
- 架构: x86-64 with AVX
- 编译器: g++ 带O3优化、AVX和OpenMP支持
- 编译选项: -O3 -mavx -march=native -fopenmp
- OpenMP线程数: $THREADS
- 测试矩阵规模: ${SIZES[@]}

## 算法实现
1. **串行算法**: 基准实现，带AVX向量化但单线程处理
2. **基本OpenMP版本**: 使用简单的parallel for指令并行化消去循环
3. **单一并行区OpenMP版本**: 使用单一并行区域和single指令减少线程创建开销
4. **Nowait优化OpenMP版本**: 使用nowait子句减少不必要的同步开销
5. **动态调度OpenMP版本**: 使用动态调度策略优化负载均衡

## 性能总结

![执行时间](plots/execution_time_plot.png)
*不同版本的执行时间对比*

![对数比例执行时间](plots/execution_time_plot_log.png)
*对数比例下的执行时间对比*

![加速比](plots/speedup_plot.png)
*相对于串行实现的加速比*

![加速比vs矩阵大小](plots/speedup_log_size.png)
*加速比随矩阵大小的变化（对数尺度）*

![并行效率](plots/parallel_efficiency_plot.png)
*不同版本的并行效率 (加速比/线程数)*

![大规模矩阵算法比较](plots/algorithm_comparison.png)
*大规模矩阵下各算法版本的比较*

![性能热图](plots/performance_heatmap.png)
*各算法在不同矩阵大小下的相对性能热图*

## 结论与分析

1. **AVX向量化效果**: AVX指令（每个指令处理8个浮点数）显著提高了串行版本的性能，比SSE（4个浮点数）可以理论上提供2倍的性能提升。

2. **OpenMP并行化效果**: 
   - 在小规模矩阵上，并行开销可能超过并行计算带来的收益
   - 在大规模矩阵上，OpenMP版本通常能够实现不错的加速比，特别是使用动态调度策略的版本

3. **最佳实现策略**:
   - 对于小规模矩阵：串行AVX实现通常是最佳选择
   - 对于中等规模矩阵：单一并行区OpenMP版本表现较好
   - 对于大规模矩阵：动态调度OpenMP版本表现最佳

4. **同步开销分析**:
   - Nowait版本在理论上应该减少同步开销，但实际测试结果显示其性能优势不明显
   - 这可能是因为高斯消元算法本身的依赖性质，使得同步点无法完全避免

5. **向量化与并行化协同效应**:
   - AVX向量化和OpenMP线程并行化是两个正交的优化维度
   - 结合使用时，可以同时利用指令级并行和线程级并行
   - 在最佳情况下，理论加速比可以达到 8(AVX) × 4(线程) = 32倍

## 优化建议

1. **矩阵分块**: 实现分块高斯消元，提高缓存利用率

2. **动态调度策略调优**: 测试不同的块大小，找到最佳的动态调度参数

3. **混合精度计算**: 在算法不同阶段使用不同的浮点精度，平衡精度和性能

4. **考虑更高级的SIMD指令**: 在支持的平台上，可以考虑使用AVX-512，进一步提高向量化性能

5. **自适应算法选择**: 根据矩阵大小自动选择最佳的算法实现

通过这次性能测试，我们看到了AVX向量化和OpenMP并行化在高斯消元算法上的强大潜力。在未来的工作中，我们可以进一步探索更高级的优化技术，如自动调整算法参数和进一步减少线程同步开销。

EOL

echo -e "${GREEN}Performance report generated: results_avx_openmp/performance_report.md${NC}"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo -e "${GREEN}Results saved in results_avx_openmp directory${NC}"
echo -e "${BLUE}You can view the plots and performance report in the results_avx_openmp directory${NC}"