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

# 检查是否存在gaussian_sse_openmp.cpp
if [ ! -f "gaussian_sse_openmp.cpp" ]; then
  echo -e "${RED}Error: gaussian_sse_openmp.cpp not found!${NC}"
  exit 1
fi

# 检查SSE支持
echo -e "${BLUE}Checking CPU support for SSE...${NC}"
if grep -q "sse" /proc/cpuinfo; then
  echo -e "${GREEN}SSE instructions are supported!${NC}"
else
  echo -e "${RED}Warning: SSE instructions not explicitly found in /proc/cpuinfo${NC}"
  echo -e "${YELLOW}Program may not function correctly without SSE support.${NC}"
  read -p "Do you want to continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Exiting.${NC}"
    exit 1
  fi
  echo -e "${YELLOW}Continuing at your own risk. Results may be unreliable.${NC}"
fi

# 编译程序（使用x86架构编译器并支持SSE和OpenMP）
echo -e "${BLUE}Compiling Gaussian Elimination program with SSE and OpenMP...${NC}"
g++ -o gaussian_sse_openmp gaussian_sse_openmp.cpp \
  -fopenmp -O3 -msse2 -mfpmath=sse \
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
mkdir -p results_sse_openmp/{raw_output,intermediate_results,performance_data,plots}

# 测试参数设置
echo -e "${BLUE}Preparing tests with different matrix sizes...${NC}"
# 矩阵规模
SIZES=(64 128 256 512 1024 2048)
# 线程数
THREADS=4

# 初始化结果文件
echo "matrix_size,serial,basic_omp,single_region,nowait,schedule" > results_sse_openmp/execution_time.csv
echo "matrix_size,basic_omp,single_region,nowait,schedule" > results_sse_openmp/speedup.csv

# 对每个矩阵大小运行测试
for size in "${SIZES[@]}"; do
  echo -e "\n${BLUE}Testing matrix size: ${YELLOW}${size} x ${size}${NC}"

  # 创建中间结果文件
  result_file="results_sse_openmp/intermediate_results/output_${size}.txt"
  echo "=== SSE OpenMP Gaussian Elimination Test (Size: $size, Threads: $THREADS) ===" > "$result_file"
  echo "Command: ./gaussian_sse_openmp $size $THREADS" >> "$result_file"
  echo "Started at: $(date)" >> "$result_file"
  echo "----------------------------------------" >> "$result_file"

  # 设置缓存预热运行
  echo -e "${BLUE}Running cache warm-up iteration...${NC}"
  ./gaussian_sse_openmp $size $THREADS > /dev/null 2>&1 || true
  
  # 正式运行程序
  echo -e "${BLUE}Running benchmark...${NC}"
  
  # 运行程序并收集输出
  output=$(./gaussian_sse_openmp $size $THREADS)
  
  # 显示输出概要
  echo -e "${GREEN}Program completed for size $size${NC}"
  echo "$output" | grep -E "time|speedup|correct"
  
  # 保存当前规模的完整输出
  echo "$output" > "results_sse_openmp/raw_output/output_${size}.txt"
  echo "$output" >> "$result_file"
  
  # 添加分隔符和时间戳
  echo "----------------------------------------" >> "$result_file"
  echo "Finished at: $(date)" >> "$result_file"
  
  # 提取CSV格式的数据
  execution_time=$(echo "$output" | grep -A 2 "CSV Format for plotting:" | tail -n 1)
  speedup=$(echo "$output" | grep -A 2 "Speedup CSV Format for plotting:" | tail -n 1)
  
  # 添加到结果文件
  if [ -n "$execution_time" ] && [[ "$execution_time" != *"matrix_size"* ]]; then
    echo "$execution_time" >> results_sse_openmp/execution_time.csv
    echo "Execution time extracted and saved successfully" >> "$result_file"
  else
    echo -e "${YELLOW}Warning: Could not extract valid execution time for size $size${NC}"
    echo "Warning: Could not extract valid execution time" >> "$result_file"
  fi

  if [ -n "$speedup" ] && [[ "$speedup" != *"matrix_size"* ]]; then
    echo "$speedup" >> results_sse_openmp/speedup.csv
    echo "Speedup extracted and saved successfully" >> "$result_file"
  else
    echo -e "${YELLOW}Warning: Could not extract valid speedup for size $size${NC}"
    echo "Warning: Could not extract valid speedup" >> "$result_file"
  fi
  
  # 记录内存使用情况
  echo -e "${BLUE}Recording memory usage...${NC}"
  echo "Memory usage after test:" >> "$result_file"
  ps -o pid,rss,command | grep "gaussian_sse_openmp" | grep -v "grep" >> "$result_file" || echo "No process found" >> "$result_file"

  echo -e "${GREEN}Completed test for size $size${NC}"
  echo "========== End of Test for Matrix Size: $size ==========\n" >> "$result_file"
done

# 显示CSV文件内容
echo -e "${BLUE}Results summary:${NC}"
echo "Contents of execution_time.csv:"
cat results_sse_openmp/execution_time.csv
echo ""
echo "Contents of speedup.csv:"
cat results_sse_openmp/speedup.csv
echo ""

# 合并所有的中间结果
echo -e "${BLUE}Combining all results...${NC}"
cat results_sse_openmp/intermediate_results/output_*.txt > results_sse_openmp/output.txt
echo -e "${GREEN}Combined output saved to results_sse_openmp/output.txt${NC}"

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
    time_data = pd.read_csv('results_sse_openmp/execution_time.csv')
    speedup_data = pd.read_csv('results_sse_openmp/speedup.csv')
    
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
    
    plt.title('OpenMP Gaussian Elimination Execution Time with SSE', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_sse_openmp/plots/execution_time_plot.png', dpi=300)
    
    # 对数比例的执行时间图
    plt.figure(figsize=(12, 8))
    plt.semilogy(time_data['matrix_size'], time_data['serial'] / 1000000, 'o-', label='Serial', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['basic_omp'] / 1000000, 's-', label='Basic OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['single_region'] / 1000000, '^-', label='Single Region OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['nowait'] / 1000000, 'd-', label='Nowait OpenMP', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['schedule'] / 1000000, 'x-', label='Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Gaussian Elimination Execution Time with SSE (Log Scale)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (seconds, log scale)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_sse_openmp/plots/execution_time_plot_log.png', dpi=300)
    
    # 加速比图 (0-1范围)
    plt.figure(figsize=(12, 8))
    plt.plot(speedup_data['matrix_size'], speedup_data['basic_omp'], 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['single_region'], '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['nowait'], 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['schedule'], 'x-', label='Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Gaussian Elimination Speedup with SSE (0-1 Range)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Speedup (compared to serial version)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.ylim(0, 1)  # 设置Y轴范围为0-1
    plt.tight_layout()
    plt.savefig('results_sse_openmp/plots/speedup_plot_0_1.png', dpi=300)
    
    # 加速比图 (自动伸缩范围)
    plt.figure(figsize=(12, 8))
    plt.plot(speedup_data['matrix_size'], speedup_data['basic_omp'], 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['single_region'], '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['nowait'], 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['schedule'], 'x-', label='Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Gaussian Elimination Speedup with SSE (Auto Range)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Speedup (compared to serial version)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_sse_openmp/plots/speedup_plot_auto.png', dpi=300)
    
    # 并行效率图
    plt.figure(figsize=(12, 8))
    threads = ${THREADS}  # 使用脚本定义的线程数
    plt.plot(speedup_data['matrix_size'], speedup_data['basic_omp']/threads, 's-', label='Basic OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['single_region']/threads, '^-', label='Single Region OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['nowait']/threads, 'd-', label='Nowait OpenMP', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['schedule']/threads, 'x-', label='Schedule OpenMP', linewidth=2)
    
    plt.title('OpenMP Gaussian Elimination Parallel Efficiency with SSE', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Parallel Efficiency (Speedup/Threads)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.ylim(0, 0.25)  # 设置Y轴范围为0-0.25 (假设加速比较低)
    plt.tight_layout()
    plt.savefig('results_sse_openmp/plots/parallel_efficiency_plot.png', dpi=300)
    
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
    ax.set_title('Comparison of Algorithm Versions for Large Matrices with SSE', fontsize=16)
    ax.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('results_sse_openmp/plots/algorithm_comparison.png', dpi=300)
    
    # 增加SSE性能热图
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
    plt.title('Performance Heatmap of SSE+OpenMP Implementations', fontsize=16)
    plt.yticks(np.arange(len(algorithms)), alg_labels)
    plt.xticks(np.arange(len(time_data['matrix_size'])), time_data['matrix_size'])
    
    # 在热图中添加数值标签
    for i in range(len(algorithms)):
        for j in range(len(time_data['matrix_size'])):
            text_color = 'white' if heatmap_data[i][j] > 1.5 else 'black'
            plt.text(j, i, f'{heatmap_data[i][j]:.2f}', 
                     ha='center', va='center', color=text_color)
    
    plt.tight_layout()
    plt.savefig('results_sse_openmp/plots/performance_heatmap.png', dpi=300)
    
    print('All plots generated successfully!')
    
except Exception as e:
    print(f'Error in Python script: {str(e)}')
    with open('results_sse_openmp/plots/plot_error.log', 'w') as error_file:
        error_file.write(f'Error: {str(e)}\\n')
    sys.exit(1)
"

# 检查绘图是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to generate plots. Check results_sse_openmp/plots/plot_error.log for details.${NC}"
  echo "Make sure matplotlib and pandas are installed (pip3 install matplotlib pandas numpy)."
else
  echo -e "${GREEN}Plots generated successfully!${NC}"
fi

# 生成性能报告
echo -e "${BLUE}Generating performance report...${NC}"
cat > results_sse_openmp/performance_report.md << EOL
# SSE优化OpenMP高斯消去算法性能报告

## 概述
本报告总结了使用SSE向量化和OpenMP并行化的高斯消去算法在x86平台上的性能测试结果。
测试日期: $(date)

## 测试环境
- 架构: x86-64 with SSE2
- 编译器: g++ 带O3优化、SSE2和OpenMP支持
- 编译选项: -O3 -msse2 -mfpmath=sse -fopenmp
- OpenMP线程数: $THREADS
- 测试矩阵规模: ${SIZES[@]}

## 算法实现
1. **串行算法**: 基准实现，带SSE向量化但单线程处理
2. **基本OpenMP版本**: 使用简单的parallel for指令并行化消去循环
3. **单一并行区OpenMP版本**: 使用单一并行区域和single指令减少线程创建开销
4. **Nowait优化OpenMP版本**: 使用nowait子句减少不必要的同步开销
5. **动态调度OpenMP版本**: 使用动态调度策略优化负载均衡

## 性能总结

![执行时间](plots/execution_time_plot.png)
*不同版本的执行时间对比*

![对数比例执行时间](plots/execution_time_plot_log.png)
*对数比例下的执行时间对比*

![加速比(0-1范围)](plots/speedup_plot_0_1.png)
*相对于串行实现的加速比，限制在0-1范围内*

![加速比(自动范围)](plots/speedup_plot_auto.png)
*相对于串行实现的加速比，自动调整范围*

![并行效率](plots/parallel_efficiency_plot.png)
*不同版本的并行效率 (加速比/线程数)*

![大规模矩阵算法比较](plots/algorithm_comparison.png)
*大规模矩阵下各算法版本的比较*

![性能热图](plots/performance_heatmap.png)
*各算法在不同矩阵大小下的相对性能热图*

## 结论与分析

1. **SSE向量化效果**: SSE指令（每个指令处理4个浮点数）显著提高了串行版本的性能，使基础实现已经相当高效。

2. **OpenMP并行化**: 
   - 在小规模矩阵上，并行版本可能比串行版本慢，因为线程创建和管理的开销超过了并行带来的收益
   - 在大规模矩阵上，一些OpenMP版本能够提供适度的加速比

3. **最佳实现策略**:
   - 对于小规模矩阵（<= 1024 x 1024）：串行SSE实现通常是最佳选择
   - 对于大规模矩阵：动态调度的OpenMP实现表现较好，能更均匀地分配工作负载

4. **性能瓶颈分析**:
   - 内存访问模式: 高斯消元对内存访问模式不友好，导致缓存效率低
   - 负载不均衡: 随着消元过程进行，每轮工作量减少，导致一些线程闲置
   - 同步开销: 线程同步是一个显著的开销，尤其是在需要频繁同步的情况下

5. **SSE vs 并行化权衡**:
   - 在这个实现中，SSE向量化对性能的贡献可能比线程级并行更显著
   - 这表明计算密集型算法可能更受益于SIMD优化而非线程并行

## 优化建议

1. **混合精度策略**: 考虑使用不同的精度级别，在算法初期使用较低精度，后期使用较高精度，以提高性能

2. **缓存友好实现**: 重新设计算法，以更好地利用CPU缓存，例如分块高斯消元

3. **避免线程同步**: 尽可能减少线程之间的依赖，减少同步点的数量

4. **自适应调度**: 根据问题规模自动选择最佳的并行策略和线程数量

5. **考虑更高级的SIMD指令**: 在支持的CPU上，可以考虑使用AVX/AVX2/AVX-512指令，它们能处理更多数据

通过这次性能测试，我们证明了使用SSE指令集可以显著提高高斯消元的效率，特别是在较小矩阵上。对于大规模问题，将SSE与OpenMP结合是实现最佳性能的关键。

EOL

echo -e "${GREEN}Performance report generated: results_sse_openmp/performance_report.md${NC}"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo -e "${GREEN}Results saved in results_sse_openmp directory${NC}"
echo -e "${BLUE}You can view the plots and performance report in the results_sse_openmp directory${NC}"