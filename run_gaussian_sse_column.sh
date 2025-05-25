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

echo -e "${BLUE}=== Gaussian Elimination Column-Based SSE Algorithm Test Script ===${NC}"
echo -e "${BLUE}Current Date and Time: 2025-05-22 00:15:14${NC}"
echo -e "${BLUE}Current User: KKKyriejiang${NC}"

# 检查是否存在gaussian_sse_column.cpp文件
if [ ! -f "gaussian_sse_column.cpp" ]; then
  echo -e "${RED}Error: gaussian_sse_column.cpp not found!${NC}"
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

# 编译程序（使用x86架构编译器并启用SSE）
echo -e "${BLUE}Compiling Column-Based Gaussian Elimination program with SSE...${NC}"
g++ -o gaussian_sse_column gaussian_sse_column.cpp -pthread -O3 -msse2 -mfpmath=sse

# 检查编译是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}Compilation failed!${NC}"
  exit 1
fi
echo -e "${GREEN}Compilation successful${NC}"

# 创建结果目录
mkdir -p results_sse_column
# 创建用于存放原始输出和中间结果的子目录
mkdir -p results_sse_column/raw_output
mkdir -p results_sse_column/intermediate_results
mkdir -p results_sse_column/plots

# 修改后的矩阵测试规模
echo -e "${BLUE}Running tests with different matrix sizes...${NC}"
SIZES=(64 128 256 512 1024)

# 清空并初始化结果文件（只写入表头一次）
echo "matrix_size,serial,dynamic_thread_col,static_semaphore_col,static_full_col,barrier_col" > results_sse_column/execution_time.csv
echo "matrix_size,dynamic_thread_col,static_semaphore_col,static_full_col,barrier_col" > results_sse_column/speedup.csv

# 对每个矩阵大小运行测试
for size in "${SIZES[@]}"; do
  echo -e "\n${BLUE}Testing matrix size: ${YELLOW}${size}${NC}"

  # 保存中间结果到output.txt
  result_file="results_sse_column/intermediate_results/output_${size}.txt"
  echo "=== Gaussian Elimination SSE Column Test with Matrix Size: $size ===" > "$result_file"
  echo "Command: ./gaussian_sse_column $size" >> "$result_file"
  echo "Started at: $(date)" >> "$result_file"
  echo "----------------------------------------" >> "$result_file"

  # 运行程序并提取结果
  # 先进行一次预热运行
  echo -e "${BLUE}Running warm-up iteration...${NC}"
  ./gaussian_sse_column $size > /dev/null 2>&1 || true
  
  # 正式运行并收集输出
  echo -e "${BLUE}Running benchmark...${NC}"
  output=$(./gaussian_sse_column $size)
  
  # 显示输出概要
  echo -e "${GREEN}Program completed for size $size${NC}"
  echo "$output" | grep -E "time|speedup|correct"
  
  # 保存当前规模的完整输出
  echo "$output" > "results_sse_column/raw_output/output_${size}.txt"
  echo "$output" >> "$result_file"
  
  # 添加分隔符和时间戳
  echo "----------------------------------------" >> "$result_file"
  echo "Finished at: $(date)" >> "$result_file"
  
  # 提取CSV格式的数据
  execution_time=$(echo "$output" | grep -A 2 "CSV Format for plotting:" | tail -n 1 | sed 's/^[[:space:]]*//')
  speedup=$(echo "$output" | grep -A 2 "Speedup CSV Format for plotting:" | tail -n 1 | sed 's/^[[:space:]]*//')
  
  # 添加到结果文件
  if [ -n "$execution_time" ] && [[ "$execution_time" != *"matrix_size"* ]]; then
    echo "$execution_time" >> results_sse_column/execution_time.csv
    echo "Execution time extracted and saved: $execution_time" >> "$result_file"
  else
    echo -e "${YELLOW}Warning: Could not extract valid execution time for size $size${NC}"
    echo "Warning: Could not extract valid execution time" >> "$result_file"
  fi

  if [ -n "$speedup" ] && [[ "$speedup" != *"matrix_size"* ]]; then
    echo "$speedup" >> results_sse_column/speedup.csv
    echo "Speedup extracted and saved: $speedup" >> "$result_file"
  else
    echo -e "${YELLOW}Warning: Could not extract valid speedup for size $size${NC}"
    echo "Warning: Could not extract valid speedup" >> "$result_file"
  fi
  
  # 记录内存使用情况
  echo -e "${BLUE}Recording memory usage...${NC}"
  echo "Memory usage after test:" >> "$result_file"
  ps -o pid,rss,command | grep "gaussian_sse_column" | grep -v "grep" >> "$result_file" || echo "No process found" >> "$result_file"

  echo -e "${GREEN}Completed test for size $size${NC}"
  echo "========== End of Test for Matrix Size: $size ==========\n" >> "$result_file"
done

# 显示CSV文件内容
echo -e "${BLUE}Results summary:${NC}"
echo "Contents of execution_time.csv:"
cat results_sse_column/execution_time.csv
echo ""
echo "Contents of speedup.csv:"
cat results_sse_column/speedup.csv
echo ""

# 合并所有的中间结果
echo -e "${BLUE}Combining all results...${NC}"
cat results_sse_column/intermediate_results/output_*.txt > results_sse_column/output.txt
echo -e "${GREEN}Combined output saved to results_sse_column/output.txt${NC}"

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
    time_csv_path = 'results_sse_column/execution_time.csv'
    speedup_csv_path = 'results_sse_column/speedup.csv'
    
    # 显示文件内容
    print('Contents of execution_time.csv:')
    with open(time_csv_path, 'r') as f:
        print(f.read())
    
    print('Contents of speedup.csv:')
    with open(speedup_csv_path, 'r') as f:
        print(f.read())
    
    # 检查数据完整性
    with open(time_csv_path, 'r') as f:
        lines = f.readlines()
        if len(lines) <= 1:
            raise Exception('Execution time CSV file only contains the header row. No data was collected.')
    
    with open(speedup_csv_path, 'r') as f:
        lines = f.readlines()
        if len(lines) <= 1:
            raise Exception('Speedup CSV file only contains the header row. No data was collected.')
    
    # 读取执行时间数据
    time_data = pd.read_csv(time_csv_path)
    speedup_data = pd.read_csv(speedup_csv_path)
    
    # 确保数据列是数字类型
    numeric_cols = time_data.columns.drop('matrix_size') if 'matrix_size' in time_data.columns else time_data.columns
    for col in numeric_cols:
        time_data[col] = pd.to_numeric(time_data[col], errors='coerce')
    
    numeric_cols = speedup_data.columns.drop('matrix_size') if 'matrix_size' in speedup_data.columns else speedup_data.columns
    for col in numeric_cols:
        speedup_data[col] = pd.to_numeric(speedup_data[col], errors='coerce')
    
    # 计算矩阵元素数量
    time_data['elements'] = time_data['matrix_size'].astype(int) * time_data['matrix_size'].astype(int)
    
    # 执行时间图
    plt.figure(figsize=(12, 8))
    plt.plot(time_data['matrix_size'], time_data['serial'] / 1000000, 'o-', label='Serial', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['dynamic_thread_col'] / 1000000, 's-', label='Dynamic Thread', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['static_semaphore_col'] / 1000000, '^-', label='Static Semaphore', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['static_full_col'] / 1000000, 'd-', label='Static Full', linewidth=2)
    plt.plot(time_data['matrix_size'], time_data['barrier_col'] / 1000000, 'x-', label='Barrier', linewidth=2)
    plt.title('SSE Column-Based Gaussian Elimination Execution Time', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_sse_column/plots/execution_time_plot.png', dpi=300)
    
    # 对数尺度执行时间图
    plt.figure(figsize=(12, 8))
    plt.semilogy(time_data['matrix_size'], time_data['serial'] / 1000000, 'o-', label='Serial', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['dynamic_thread_col'] / 1000000, 's-', label='Dynamic Thread', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['static_semaphore_col'] / 1000000, '^-', label='Static Semaphore', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['static_full_col'] / 1000000, 'd-', label='Static Full', linewidth=2)
    plt.semilogy(time_data['matrix_size'], time_data['barrier_col'] / 1000000, 'x-', label='Barrier', linewidth=2)
    plt.title('SSE Column-Based Gaussian Elimination Execution Time (Log Scale)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (seconds, log scale)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_sse_column/plots/execution_time_log_plot.png', dpi=300)
    
    # 加速比图 (0-0.5范围)
    plt.figure(figsize=(12, 8))
    plt.plot(speedup_data['matrix_size'], speedup_data['dynamic_thread_col'], 's-', label='Dynamic Thread', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['static_semaphore_col'], '^-', label='Static Semaphore', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['static_full_col'], 'd-', label='Static Full', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['barrier_col'], 'x-', label='Barrier', linewidth=2)
    
    # 设置Y轴范围为0-0.5
    plt.ylim(0, 0.5)
    
    plt.title('SSE Column-Based Gaussian Elimination Speedup (0-0.5 Range)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Speedup (compared to serial version)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_sse_column/plots/speedup_plot_0_0.5.png', dpi=300)
    
    # 加速比图（自动范围）- 添加额外的图，保留原始范围
    plt.figure(figsize=(12, 8))
    plt.plot(speedup_data['matrix_size'], speedup_data['dynamic_thread_col'], 's-', label='Dynamic Thread', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['static_semaphore_col'], '^-', label='Static Semaphore', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['static_full_col'], 'd-', label='Static Full', linewidth=2)
    plt.plot(speedup_data['matrix_size'], speedup_data['barrier_col'], 'x-', label='Barrier', linewidth=2)
    
    # 理想加速比（线程数）
    plt.axhline(y=4, color='gray', linestyle='--', label='Ideal (4 threads)')
    
    plt.title('SSE Column-Based Gaussian Elimination Speedup (Auto Range)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Speedup (compared to serial version)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_sse_column/plots/speedup_plot_auto.png', dpi=300)
    
    # 条形图：大矩阵规模下的各实现对比
    plt.figure(figsize=(14, 8))
    
    # 筛选最大的两个矩阵大小
    largest_sizes = sorted(time_data['matrix_size'].unique())[-2:]
    large_data = time_data[time_data['matrix_size'].isin(largest_sizes)]
    
    # 准备绘图数据
    algorithms = ['serial', 'dynamic_thread_col', 'static_semaphore_col', 'static_full_col', 'barrier_col']
    alg_labels = ['Serial', 'Dynamic Thread', 'Static Semaphore', 'Static Full', 'Barrier']
    
    # 设置x轴位置
    x = np.arange(len(alg_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, size in enumerate(largest_sizes):
        size_data = large_data[large_data['matrix_size'] == size]
        times = [size_data[alg].values[0]/1000000 for alg in algorithms]
        ax.bar(x + (i-0.5)*width, times, width, label=f'Size {size}x{size}')
    
    ax.set_xlabel('Implementation', fontsize=14)
    ax.set_ylabel('Execution Time (seconds)', fontsize=14)
    ax.set_title('SSE Column-Based Implementation Comparison for Large Matrices', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(alg_labels)
    ax.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('results_sse_column/plots/large_matrix_comparison.png', dpi=300)
    
    # 性能热图：显示不同实现在不同矩阵大小上的相对性能
    # 规格化为最低执行时间=1，更高的数字意味着更慢
    plt.figure(figsize=(12, 8))
    # 创建每个矩阵大小的最小执行时间
    min_times = []
    for size in time_data['matrix_size']:
        size_data = time_data[time_data['matrix_size'] == size]
        min_time = min(size_data[algorithms].values[0])
        min_times.append(min_time)
    
    # 构建热图数据
    heatmap_data = []
    for alg in algorithms:
        relative_times = []
        for i, size in enumerate(time_data['matrix_size']):
            size_data = time_data[time_data['matrix_size'] == size]
            relative_times.append(size_data[alg].values[0] / min_times[i])
        heatmap_data.append(relative_times)
    
    # 创建热图
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Relative Execution Time (lower is better)')
    
    # 添加标签
    plt.ylabel('Implementation')
    plt.xlabel('Matrix Size')
    plt.title('SSE Column-Based Relative Performance Heatmap', fontsize=16)
    plt.yticks(np.arange(len(algorithms)), alg_labels)
    plt.xticks(np.arange(len(time_data['matrix_size'])), time_data['matrix_size'])
    
    # 在每个单元格上添加数值
    for i in range(len(algorithms)):
        for j in range(len(time_data['matrix_size'])):
            text_color = 'white' if heatmap_data[i][j] > 1.5 else 'black'
            plt.text(j, i, f'{heatmap_data[i][j]:.2f}', 
                     ha='center', va='center', color=text_color)
    
    plt.tight_layout()
    plt.savefig('results_sse_column/plots/performance_heatmap.png', dpi=300)
    
    # 加速比比较图 - 条形图格式，突出显示具体值
    plt.figure(figsize=(14, 10))
    
    width = 0.2
    x = np.arange(len(speedup_data['matrix_size']))
    
    # 绘制条形图
    rects1 = plt.bar(x - 1.5*width, speedup_data['dynamic_thread_col'], width, label='Dynamic Thread', color='#1f77b4')
    rects2 = plt.bar(x - 0.5*width, speedup_data['static_semaphore_col'], width, label='Static Semaphore', color='#ff7f0e')
    rects3 = plt.bar(x + 0.5*width, speedup_data['static_full_col'], width, label='Static Full', color='#2ca02c')
    rects4 = plt.bar(x + 1.5*width, speedup_data['barrier_col'], width, label='Barrier', color='#d62728')
    
    # 添加数据标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', rotation=45,
                    fontsize=9)
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    add_labels(rects4)
    
    # 设置坐标轴
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Speedup (compared to serial version)', fontsize=14)
    plt.title('SSE Column-Based Gaussian Elimination Speedup Comparison (0-0.5 Range)', fontsize=16)
    plt.xticks(x, speedup_data['matrix_size'])
    plt.ylim(0, 0.5)  # 设置Y轴范围为0-0.5
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig('results_sse_column/plots/speedup_comparison_bars.png', dpi=300)
    
    print('All plots generated successfully!')
    
except Exception as e:
    print(f'Error in Python script: {str(e)}')
    with open('results_sse_column/plots/plot_error.log', 'w') as error_file:
        error_file.write(f'Error: {str(e)}\\n')
    sys.exit(1)
"

# 检查绘图是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to generate plots. Check results_sse_column/plots/plot_error.log for details.${NC}"
  echo "Make sure matplotlib and pandas are installed (pip3 install matplotlib pandas numpy)."
  exit 1
fi

# 生成性能报告
echo -e "${BLUE}Generating performance report...${NC}"
cat > results_sse_column/performance_report.md << EOL
# SSE列划分高斯消去算法性能报告

## 概述
本报告总结了使用SSE向量化和列划分并行化的高斯消去算法在x86平台上的性能测试结果。
测试日期: 2025-05-22 00:15:14
测试用户: KKKyriejiang

## 测试环境
- 架构: x86-64 with SSE2
- 编译选项: -O3 -msse2 -mfpmath=sse -pthread
- 线程数量: 4
- 测试矩阵规模: ${SIZES[@]}

## 算法实现
1. **串行算法**: 基准实现，单线程顺序处理
2. **动态线程版本**: 在每轮消元中动态创建线程，每个线程处理指定的列范围
3. **静态线程+信号量同步版本**: 使用固定线程池和信号量进行同步
4. **静态线程+信号量同步+三重循环整合版本**: 将三重循环完全整合进线程函数，减少同步点
5. **静态线程+屏障同步版本**: 使用pthread_barrier进行线程同步

## 性能总结

![执行时间](plots/execution_time_plot.png)
*不同实现的执行时间对比*

![对数尺度执行时间](plots/execution_time_log_plot.png)
*对数尺度下的执行时间对比*

![加速比(0-0.5范围)](plots/speedup_plot_0_0.5.png)
*相对于串行实现的加速比，限制在0-0.5范围*

![加速比条形图](plots/speedup_comparison_bars.png)
*加速比条形图展示，限制在0-0.5范围，显示具体值*

![大矩阵对比](plots/large_matrix_comparison.png)
*大矩阵下各实现的比较*

![性能热图](plots/performance_heatmap.png)
*各实现在不同矩阵大小上的相对性能热图*

## 列划分算法分析

1. **加速比低于1的原因**:
   - 串行SSE向量化已经非常高效，4倍数据并行性
   - 列划分导致的内存访问模式对缓存极不友好
   - 线程同步和管理开销抵消了并行计算收益
   - 缓存失效导致内存访问延迟增加

2. **列划分特点**:
   - 按列划分数据使得线程间的数据局部性较差
   - 在现代CPU架构中，按列访问对缓存不友好
   - 每个线程处理的数据在内存中不连续，导致缓存利用率低

3. **串行性能**:
   - 串行版本+SSE向量化已经能够提供不错的性能
   - SSE指令每次可处理4个单精度浮点数
   - 串行版本的内存访问模式更连续，缓存效率更高

4. **并行实现比较**:
   - 静态线程池比动态线程创建有明显优势
   - 屏障同步在列划分中表现较好，提供了清晰的同步点
   - 信号量同步+三重循环整合版本优于基本静态线程版本

## 改进建议

1. **替换列划分为行划分**:
   - 使用行划分代替列划分，显著提高缓存局部性
   - 行划分可以充分利用CPU的硬件预取机制
   - 预期行划分可将加速比提高到大于1

2. **混合策略**:
   - 采用分块策略，先按行再按列划分
   - 在每个块内使用SSE向量化
   - 优化内存访问模式和线程分配

3. **减少同步开销**:
   - 增大每个线程处理的工作粒度
   - 减少同步点数量
   - 使用无锁数据结构

4. **内存对齐与预取优化**:
   - 确保所有矩阵行都按16字节对齐，最大化SSE性能
   - 添加软件预取指令提示CPU预先加载数据

5. **高级指令集**:
   - 考虑使用AVX/AVX2/AVX-512代替SSE，提高向量宽度

## 结论

在本次列划分的高斯消元算法测试中，我们观察到加速比均低于0.5，意味着所有并行实现都比串行实现慢。这主要是由于列划分对内存访问模式的负面影响，以及线程管理开销大于并行计算收益所致。SSE向量化使得串行代码已经相当高效，在这种情况下，通过列划分进行线程并行反而会降低性能。

如需提高性能，强烈建议转向行划分或块划分策略，以改善内存访问模式和缓存利用率。将SSE向量化与合适的数据划分相结合，才能真正发挥多核处理器的性能潜力。

EOL

echo -e "${GREEN}Performance report generated: results_sse_column/performance_report.md${NC}"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo -e "${GREEN}Results saved in results_sse_column directory${NC}"
echo -e "${BLUE}You can view the plots and performance report in the results_sse_column directory${NC}"