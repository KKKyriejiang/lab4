#!/bin/bash

# Define color settings
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Gaussian Elimination Algorithm Test Script (KKKyriejiang) ===${NC}"
echo -e "${BLUE}Current Date and Time: $(date -u "+%Y-%m-%d %H:%M:%S")${NC}"
echo -e "${BLUE}Current User: KKKyriejiang${NC}"

# Check for AVX support
echo -e "${BLUE}Checking CPU support for AVX...${NC}"
if grep -q "avx" /proc/cpuinfo; then
  echo -e "${GREEN}AVX instructions are supported!${NC}"
else
  echo -e "${RED}This CPU does not support AVX instructions!${NC}"
  echo -e "${YELLOW}The program requires a CPU with AVX support to run correctly.${NC}"
  read -p "Do you want to continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Exiting.${NC}"
    exit 1
  fi
  echo -e "${YELLOW}Continuing at your own risk. Results may be unreliable.${NC}"
fi

# Directory structure - use a timestamp to avoid overwriting previous results
TIMESTAMP=$(date -u "+%Y%m%d_%H%M%S")
RESULTS_DIR="results_avx_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}/raw_output
mkdir -p ${RESULTS_DIR}/plots
mkdir -p ${RESULTS_DIR}/report

# Compile the program
echo -e "${BLUE}Compiling Gaussian Elimination program (AVX optimized version)...${NC}"
g++ -O3 -mavx -pthread -o gaussian_avx_test gaussian_avx.cpp

# Check if compilation was successful
if [ $? -ne 0 ]; then
  echo -e "${RED}Compilation failed!${NC}"
  exit 1
fi
echo -e "${GREEN}Compilation successful!${NC}"

# Initialize result files
echo "matrix_size,thread_count,serial,dynamic_thread,static_semaphore,static_full,barrier" > ${RESULTS_DIR}/execution_time.csv
echo "matrix_size,thread_count,dynamic_thread,static_semaphore,static_full,barrier" > ${RESULTS_DIR}/speedup.csv

# Define test sizes and thread counts
SIZES=(100 200 500 1000)
THREADS=(1 2 4 8 16)

# Run tests with each matrix size and thread count
for size in "${SIZES[@]}"; do
  echo -e "\n${BLUE}Testing matrix size: ${YELLOW}${size} x ${size}${NC}"
  
  for thread_count in "${THREADS[@]}"; do
    echo -e "\n${YELLOW}  Testing with ${thread_count} threads${NC}"
    
    # Set timeout based on matrix size
    if [ $size -ge 1000 ]; then
      TIMEOUT=600  # 10 minutes for large matrices
    else
      TIMEOUT=300  # 5 minutes for smaller matrices
    fi
    
    echo -e "${BLUE}Running: ./gaussian_avx_test $size $thread_count${NC}"
    
    # Run the program with timeout to prevent hanging on large matrices
    timeout $TIMEOUT ./gaussian_avx_test $size $thread_count > ${RESULTS_DIR}/raw_output/output_${size}_${thread_count}.txt
    
    # Check if program timed out
    if [ $? -eq 124 ]; then
      echo -e "${RED}Program timed out after $TIMEOUT seconds for size $size, threads $thread_count. Skipping.${NC}"
      continue
    fi
    
    # Extract CSV format data directly from the output
    # Find the line after "CSV Format for plotting:" and the line containing data
    EXEC_LINE=$(grep -A 2 "CSV Format for plotting:" ${RESULTS_DIR}/raw_output/output_${size}_${thread_count}.txt | tail -1)
    
    # Find the line after "Speedup CSV Format for plotting:" and the line containing data
    SPEEDUP_LINE=$(grep -A 2 "Speedup CSV Format for plotting:" ${RESULTS_DIR}/raw_output/output_${size}_${thread_count}.txt | tail -1)
    
    # Add data to CSV files if found
    if [[ "$EXEC_LINE" == "$size"* ]]; then
      echo -e "${GREEN}Found execution time data: $EXEC_LINE${NC}"
      echo "$EXEC_LINE" >> ${RESULTS_DIR}/execution_time.csv
    else
      echo -e "${RED}Could not find execution time data for size $size, threads $thread_count${NC}"
    fi
    
    if [[ "$SPEEDUP_LINE" == "$size"* ]]; then
      echo -e "${GREEN}Found speedup data: $SPEEDUP_LINE${NC}"
      echo "$SPEEDUP_LINE" >> ${RESULTS_DIR}/speedup.csv
    else
      echo -e "${RED}Could not find speedup data for size $size, threads $thread_count${NC}"
    fi
    
    # Show execution times and speedups for this size
    echo -e "${BLUE}Execution times for size $size with $thread_count threads:${NC}"
    grep "execution time:" ${RESULTS_DIR}/raw_output/output_${size}_${thread_count}.txt
    
    echo -e "${BLUE}Speedups for size $size with $thread_count threads:${NC}"
    grep "speedup:" ${RESULTS_DIR}/raw_output/output_${size}_${thread_count}.txt
  done
  
  echo -e "${GREEN}Test completed for matrix size $size with all thread configurations${NC}"
  echo "=========================================="
done

# Check if we have enough data to generate plots
csv_lines=$(wc -l < ${RESULTS_DIR}/execution_time.csv)
if [ "$csv_lines" -le 1 ]; then
  echo -e "${RED}Error: No execution time data collected. Cannot generate plots.${NC}"
  exit 1
fi

# Generate enhanced plots
echo -e "${BLUE}Generating enhanced plots with improved visibility...${NC}"
python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# Read data
time_csv_path = '${RESULTS_DIR}/execution_time.csv'
speedup_csv_path = '${RESULTS_DIR}/speedup.csv'

time_data = pd.read_csv(time_csv_path)
speedup_data = pd.read_csv(speedup_csv_path)

print('Execution time data:')
print(time_data)
print('\\nSpeedup data:')
print(speedup_data)

# Define colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot 1: 线程数对性能的影响 (对于每个矩阵大小)
for method in ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']:
    plt.figure(figsize=(12, 8))
    
    # 按矩阵大小分组
    matrix_sizes = time_data['matrix_size'].unique()
    for i, size in enumerate(sorted(matrix_sizes)):
        subset = time_data[time_data['matrix_size'] == size]
        # 按线程数排序
        subset = subset.sort_values('thread_count')
        plt.plot(subset['thread_count'], subset[method]/1000000, 
                 marker='o', linewidth=2, label=f'Matrix size: {size}x{size}')
    
    plt.title(f'Impact of Thread Count on {method.replace(\"_\", \" \").title()} Performance', fontsize=16)
    plt.xlabel('Thread Count', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(f'${RESULTS_DIR}/plots/thread_impact_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot 2: 加速比与线程数的关系 (对于每个矩阵大小)
for method in ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']:
    plt.figure(figsize=(12, 8))
    
    # 按矩阵大小分组
    matrix_sizes = speedup_data['matrix_size'].unique()
    for i, size in enumerate(sorted(matrix_sizes)):
        subset = speedup_data[speedup_data['matrix_size'] == size]
        # 按线程数排序
        subset = subset.sort_values('thread_count')
        plt.plot(subset['thread_count'], subset[method], 
                 marker='o', linewidth=2, label=f'Matrix size: {size}x{size}')
    
    # 添加理想加速比参考线
    max_threads = speedup_data['thread_count'].max()
    plt.plot(range(1, max_threads+1), range(1, max_threads+1), 'k--', 
             label='Ideal speedup', alpha=0.7)
    
    plt.title(f'Speedup vs Thread Count ({method.replace(\"_\", \" \").title()})', fontsize=16)
    plt.xlabel('Thread Count', fontsize=14)
    plt.ylabel('Speedup relative to serial version', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(f'${RESULTS_DIR}/plots/speedup_vs_threads_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot 3: 矩阵大小与线程数的热图 (每种方法单独一个)
for method in ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']:
    plt.figure(figsize=(10, 8))
    
    # 准备热图数据
    pivot_data = time_data.pivot_table(values=method, index='matrix_size', columns='thread_count')
    
    # 创建热图
    im = plt.imshow(pivot_data, cmap='viridis_r')
    cbar = plt.colorbar(im)
    cbar.set_label('Execution Time (microseconds)', fontsize=12)
    
    # 设置标签
    plt.title(f'Performance Heatmap: {method.replace(\"_\", \" \").title()}', fontsize=16)
    plt.xlabel('Thread Count', fontsize=14)
    plt.ylabel('Matrix Size', fontsize=14)
    
    # 设置刻度标签
    plt.xticks(np.arange(len(pivot_data.columns)), pivot_data.columns)
    plt.yticks(np.arange(len(pivot_data.index)), pivot_data.index)
    
    # 添加数值标签
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = plt.text(j, i, int(pivot_data.iloc[i, j]/1000000),
                          ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.savefig(f'${RESULTS_DIR}/plots/heatmap_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot 4: 线程效率分析 (efficiency = speedup/thread_count)
plt.figure(figsize=(12, 8))

# 计算效率
for method in ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']:
    plt.figure(figsize=(12, 8))
    
    # 按矩阵大小分组
    matrix_sizes = speedup_data['matrix_size'].unique()
    for i, size in enumerate(sorted(matrix_sizes)):
        subset = speedup_data[speedup_data['matrix_size'] == size]
        # 按线程数排序
        subset = subset.sort_values('thread_count')
        # 计算效率 = 加速比/线程数
        efficiency = subset[method] / subset['thread_count']
        plt.plot(subset['thread_count'], efficiency, 
                 marker='o', linewidth=2, label=f'Matrix size: {size}x{size}')
    
    plt.title(f'Thread Efficiency for {method.replace(\"_\", \" \").title()}', fontsize=16)
    plt.xlabel('Thread Count', fontsize=14)
    plt.ylabel('Efficiency (Speedup/Thread Count)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(f'${RESULTS_DIR}/plots/efficiency_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 添加原有的绘图代码
# Plot 1: Enhanced Execution Time Comparison with Multiple Views
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# 选取每个矩阵大小的4线程数据进行原有比较
thread4_data = time_data[time_data['thread_count'] == 4].sort_values('matrix_size')

# Top-left: Regular scale with all implementations
axes[0, 0].plot(thread4_data['matrix_size'], thread4_data['serial'] / 1000000, 'o-', color=colors[0], label='Serial', linewidth=3, markersize=10)
axes[0, 0].plot(thread4_data['matrix_size'], thread4_data['dynamic_thread'] / 1000000, 's-', color=colors[1], label='Dynamic Thread', linewidth=3, markersize=10)
axes[0, 0].plot(thread4_data['matrix_size'], thread4_data['static_semaphore'] / 1000000, '^-', color=colors[2], label='Static Semaphore', linewidth=3, markersize=10)
axes[0, 0].plot(thread4_data['matrix_size'], thread4_data['static_full'] / 1000000, 'd-', color=colors[3], label='Static Full', linewidth=3, markersize=10)
axes[0, 0].plot(thread4_data['matrix_size'], thread4_data['barrier'] / 1000000, 'x-', color=colors[4], label='Barrier', linewidth=3, markersize=10)
axes[0, 0].set_title('All Implementations (4 Threads)', fontsize=16)
axes[0, 0].set_xlabel('Matrix Size', fontsize=14)
axes[0, 0].set_ylabel('Execution Time (seconds)', fontsize=14)
axes[0, 0].grid(True)
axes[0, 0].legend(fontsize=12)

# Top-right: Log scale for all implementations to show extreme differences
axes[0, 1].semilogy(thread4_data['matrix_size'], thread4_data['serial'] / 1000000, 'o-', color=colors[0], label='Serial', linewidth=3, markersize=10)
axes[0, 1].semilogy(thread4_data['matrix_size'], thread4_data['dynamic_thread'] / 1000000, 's-', color=colors[1], label='Dynamic Thread', linewidth=3, markersize=10)
axes[0, 1].semilogy(thread4_data['matrix_size'], thread4_data['static_semaphore'] / 1000000, '^-', color=colors[2], label='Static Semaphore', linewidth=3, markersize=10)
axes[0, 1].semilogy(thread4_data['matrix_size'], thread4_data['static_full'] / 1000000, 'd-', color=colors[3], label='Static Full', linewidth=3, markersize=10)
axes[0, 1].semilogy(thread4_data['matrix_size'], thread4_data['barrier'] / 1000000, 'x-', color=colors[4], label='Barrier', linewidth=3, markersize=10)
axes[0, 1].set_title('Log Scale - All Implementations (4 Threads)', fontsize=16)
axes[0, 1].set_xlabel('Matrix Size', fontsize=14)
axes[0, 1].set_ylabel('Execution Time (seconds, log scale)', fontsize=14)
axes[0, 1].grid(True)
axes[0, 1].legend(fontsize=12)

# Bottom-left: Focus on serial and parallel implementations (excluding dynamic thread)
axes[1, 0].plot(thread4_data['matrix_size'], thread4_data['serial'] / 1000000, 'o-', color=colors[0], label='Serial', linewidth=3, markersize=10)
axes[1, 0].plot(thread4_data['matrix_size'], thread4_data['static_semaphore'] / 1000000, '^-', color=colors[2], label='Static Semaphore', linewidth=3, markersize=10)
axes[1, 0].plot(thread4_data['matrix_size'], thread4_data['static_full'] / 1000000, 'd-', color=colors[3], label='Static Full', linewidth=3, markersize=10)
axes[1, 0].plot(thread4_data['matrix_size'], thread4_data['barrier'] / 1000000, 'x-', color=colors[4], label='Barrier', linewidth=3, markersize=10)
axes[1, 0].set_title('Serial vs. Efficient Parallel Implementations (4 Threads)', fontsize=16)
axes[1, 0].set_xlabel('Matrix Size', fontsize=14)
axes[1, 0].set_ylabel('Execution Time (seconds)', fontsize=14)
axes[1, 0].grid(True)
axes[1, 0].legend(fontsize=12)

# Bottom-right: Focus only on the three parallel implementations to highlight differences
axes[1, 1].plot(thread4_data['matrix_size'], thread4_data['static_semaphore'] / 1000000, '^-', color=colors[2], label='Static Semaphore', linewidth=3, markersize=10)
axes[1, 1].plot(thread4_data['matrix_size'], thread4_data['static_full'] / 1000000, 'd-', color=colors[3], label='Static Full', linewidth=3, markersize=10)
axes[1, 1].plot(thread4_data['matrix_size'], thread4_data['barrier'] / 1000000, 'x-', color=colors[4], label='Barrier', linewidth=3, markersize=10)
axes[1, 1].set_title('Comparison of Efficient Parallel Implementations (4 Threads)', fontsize=16)
axes[1, 1].set_xlabel('Matrix Size', fontsize=14)
axes[1, 1].set_ylabel('Execution Time (seconds)', fontsize=14)
axes[1, 1].grid(True)
axes[1, 1].legend(fontsize=12)

plt.tight_layout()
plt.suptitle('Enhanced Execution Time Comparison (AVX Vectorization)', fontsize=20, y=1.02)
plt.savefig('${RESULTS_DIR}/plots/enhanced_execution_time.png', dpi=300, bbox_inches='tight')

print('All enhanced plots generated successfully!')
"

# Check if plotting was successful
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to generate plots. Make sure matplotlib and pandas are installed.${NC}"
  
  # Try to install missing dependencies and retry
  echo -e "${YELLOW}Attempting to install required packages...${NC}"
  pip3 install matplotlib pandas numpy 2>/dev/null
  
  # Retry plotting with a simplified script
  echo -e "${YELLOW}Retrying with simplified plotting...${NC}"
  python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data
time_data = pd.read_csv('${RESULTS_DIR}/execution_time.csv')
speedup_data = pd.read_csv('${RESULTS_DIR}/speedup.csv')

# Create simple speedup plot with 0-1 Y-axis range
plt.figure(figsize=(10, 6))
thread4_data = speedup_data[speedup_data['thread_count'] == 4].sort_values('matrix_size')
for col in thread4_data.columns:
    if col not in ['matrix_size', 'thread_count']:
        plt.plot(thread4_data['matrix_size'], thread4_data[col], '-o', label=col)

plt.title('Speedup Comparison (0-1 Range) - 4 Threads')
plt.xlabel('Matrix Size')
plt.ylabel('Speedup (relative to serial)')
plt.grid(True)
plt.legend()
plt.ylim(0, 1)
plt.savefig('${RESULTS_DIR}/plots/basic_speedup_0_1.png', dpi=300)
print('Basic plot with 0-1 Y-axis range created.')
"
  if [ $? -ne 0 ]; then
    echo -e "${RED}Simplified plotting also failed. Please check Python installation.${NC}"
    exit 1
  fi
fi
echo -e "${GREEN}Enhanced plots saved in ${RESULTS_DIR}/plots directory.${NC}"

# Generate performance report
echo -e "${BLUE}Generating performance report...${NC}"

cat > ${RESULTS_DIR}/report/performance_report.md << EOL
# Gaussian Elimination Algorithm Performance Report (AVX)

## Overview
This report summarizes the performance testing results of the AVX-optimized Gaussian elimination algorithm on x86 platform.
Test date: $(date -u "+%Y-%m-%d %H:%M:%S")

## Test Environment
- Architecture: x86-64 with AVX support
- Compiler: G++ with O3 and AVX optimization flags
- Thread counts tested: ${THREADS[@]}
- Test sizes: ${SIZES[@]} (matrix sizes)
- SIMD width: 8 floating point elements per vector (256 bits)

## Algorithm Implementations
1. **Serial Algorithm**: Baseline implementation using AVX vectorization
2. **Dynamic Thread Version**: Creates threads dynamically for each elimination round
3. **Static Thread + Semaphore Synchronization**: Fixed thread pool with semaphore synchronization
4. **Static Thread + Semaphore + Three-level Loop**: All loops within thread functions
5. **Static Thread + Barrier Synchronization**: Uses barrier synchronization

## Thread Scaling Performance

### Dynamic Thread Version
![Thread Impact](../plots/thread_impact_dynamic_thread.png)
*Impact of thread count on Dynamic Thread implementation execution time*

### Static Semaphore Version
![Thread Impact](../plots/thread_impact_static_semaphore.png)
*Impact of thread count on Static Semaphore implementation execution time*

### Static Full Version
![Thread Impact](../plots/thread_impact_static_full.png)
*Impact of thread count on Static Full implementation execution time*

### Barrier Synchronization Version
![Thread Impact](../plots/thread_impact_barrier.png)
*Impact of thread count on Barrier Synchronization implementation execution time*

## Thread Speedup Analysis

### Dynamic Thread Version
![Speedup vs Threads](../plots/speedup_vs_threads_dynamic_thread.png)
*How speedup scales with thread count for Dynamic Thread implementation*

### Static Semaphore Version
![Speedup vs Threads](../plots/speedup_vs_threads_static_semaphore.png)
*How speedup scales with thread count for Static Semaphore implementation*

### Static Full Version
![Speedup vs Threads](../plots/speedup_vs_threads_static_full.png)
*How speedup scales with thread count for Static Full implementation*

### Barrier Synchronization Version
![Speedup vs Threads](../plots/speedup_vs_threads_barrier.png)
*How speedup scales with thread count for Barrier Synchronization implementation*

## Thread Efficiency Analysis

### Dynamic Thread Version
![Efficiency](../plots/efficiency_dynamic_thread.png)
*Thread efficiency (speedup/thread count) for Dynamic Thread implementation*

### Static Semaphore Version
![Efficiency](../plots/efficiency_static_semaphore.png)
*Thread efficiency (speedup/thread count) for Static Semaphore implementation*

### Static Full Version
![Efficiency](../plots/efficiency_static_full.png)
*Thread efficiency (speedup/thread count) for Static Full implementation*

### Barrier Synchronization Version
![Efficiency](../plots/efficiency_barrier.png)
*Thread efficiency (speedup/thread count) for Barrier Synchronization implementation*

## Performance Heatmaps

### Dynamic Thread Version
![Heatmap](../plots/heatmap_dynamic_thread.png)
*Performance heatmap showing execution times for different matrix sizes and thread counts*

### Static Semaphore Version
![Heatmap](../plots/heatmap_static_semaphore.png)
*Performance heatmap showing execution times for different matrix sizes and thread counts*

### Static Full Version
![Heatmap](../plots/heatmap_static_full.png)
*Performance heatmap showing execution times for different matrix sizes and thread counts*

### Barrier Synchronization Version
![Heatmap](../plots/heatmap_barrier.png)
*Performance heatmap showing execution times for different matrix sizes and thread counts*

## Traditional Execution Time Comparison
![Enhanced Execution Time](../plots/enhanced_execution_time.png)
*Multiple views of execution time data for 4 threads across different matrix sizes*

## Performance Data

### Execution Time Sample (microseconds)
\`\`\`
$(head -5 ${RESULTS_DIR}/execution_time.csv)
...
\`\`\`

### Speedup Sample (relative to serial)
\`\`\`
$(head -5 ${RESULTS_DIR}/speedup.csv)
...
\`\`\`

## Key Observations

1. **Thread Scaling Behavior**: Analysis of how each implementation scales with increasing thread count.

2. **Implementation Differences**: Comparison of different synchronization methods across thread counts.

3. **Efficiency Analysis**: How efficiently each implementation utilizes the available threads.

4. **Matrix Size Impact**: How matrix size influences the benefits of multithreading.

5. **Optimal Thread Count**: Identification of optimal thread counts for different implementations and matrix sizes.

## Conclusions

Based on the test results, we can draw the following conclusions:

1. **[Summary of which implementation performs best with which thread count]**

2. **[Summary of when parallelism provides benefits over serial execution]**

3. **[Summary of synchronization method impacts]**

4. **[Recommendations for optimal thread count selection]**

EOL

echo -e "${GREEN}Performance report generated: ${RESULTS_DIR}/report/performance_report.md${NC}"
echo -e "${GREEN}All tests completed! Results saved in ${RESULTS_DIR} directory${NC}"
echo -e "${BLUE}Run end time: $(date -u "+%Y-%m-%d %H:%M:%S")${NC}"
echo -e "${BLUE}Thank you for using the Gaussian Elimination Algorithm Test Script - by KKKyriejiang${NC}"