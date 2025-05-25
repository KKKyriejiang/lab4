#!/bin/bash

# 编译程序（使用 ARM 架构交叉编译器）
echo "Cross-compiling Cache-Optimized Row-Based Gaussian Elimination program for ARM..."
aarch64-linux-gnu-g++ -static -o gaussian_elimination_row_cache gaussian_elimination_row_cache.cpp -lpthread -O3

# 检查编译是否成功
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

# 创建结果目录
mkdir -p results_row_cache
# 创建用于存放原始输出和中间结果的子目录
mkdir -p results_row_cache/raw_output
mkdir -p results_row_cache/intermediate_results

# 矩阵测试规模
echo "Running tests with different matrix sizes..."
SIZES=(16 32 64 128)

# 清空并初始化结果文件（只写入表头一次）
echo "matrix_size,serial,serial_blocked,dynamic_thread,static_semaphore,static_full,barrier" > results_row_cache/execution_time.csv
echo "matrix_size,serial_blocked,dynamic_thread,static_semaphore,static_full,barrier" > results_row_cache/speedup.csv

# 对每个矩阵大小运行测试（通过 QEMU 执行 ARM 可执行文件）
for size in "${SIZES[@]}"; do
  echo "Testing matrix size: $size"

  # 保存中间结果到output.txt
  echo "=== Cache-Optimized Gaussian Elimination Test (Row-Based) with Matrix Size: $size ===" > "results_row_cache/intermediate_results/output_${size}.txt"
  echo "Command: qemu-aarch64 ./gaussian_elimination_row_cache $size" >> "results_row_cache/intermediate_results/output_${size}.txt"
  echo "Started at: $(date)" >> "results_row_cache/intermediate_results/output_${size}.txt"
  echo "----------------------------------------" >> "results_row_cache/intermediate_results/output_${size}.txt"

  # 运行程序并提取结果
  output=$(qemu-aarch64 ./gaussian_elimination_row_cache $size)
  
  # 显示输出来检查是否正确执行
  echo "Program output for size $size:"
  echo "$output"

  # 保存当前规模的完整输出到原始输出目录
  echo "Saving raw output for size $size..."
  echo "$output" > "results_row_cache/raw_output/output_${size}.txt"
  
  # 同时将输出添加到中间结果文件
  echo "$output" >> "results_row_cache/intermediate_results/output_${size}.txt"
  
  # 为中间结果文件添加分隔符和时间戳
  echo "----------------------------------------" >> "results_row_cache/intermediate_results/output_${size}.txt"
  echo "Finished at: $(date)" >> "results_row_cache/intermediate_results/output_${size}.txt"
  
  # 提取CSV格式的数据（修改这里以确保正确提取）
  # 使用更精确的grep和sed来提取CSV行
  execution_time=$(echo "$output" | grep -A 2 "CSV Format for plotting:" | tail -n 1 | sed 's/^[[:space:]]*//')
  speedup=$(echo "$output" | grep -A 2 "Speedup CSV Format for plotting:" | tail -n 1 | sed 's/^[[:space:]]*//')
  
  echo "Extracted execution time line: '$execution_time'"
  echo "Extracted speedup line: '$speedup'"

  # 添加到结果文件
  # 检查提取的数据是否为空，避免写入空行到CSV
  if [ -n "$execution_time" ] && [[ "$execution_time" != *"matrix_size"* ]]; then
    echo "$execution_time" >> results_row_cache/execution_time.csv
    echo "Execution time extracted and saved: $execution_time" >> "results_row_cache/intermediate_results/output_${size}.txt"
  else
    echo "Warning: Could not extract valid execution time for size $size. Raw output saved in results_row_cache/raw_output/output_${size}.txt"
    echo "Warning: Could not extract valid execution time" >> "results_row_cache/intermediate_results/output_${size}.txt"
  fi

  if [ -n "$speedup" ] && [[ "$speedup" != *"matrix_size"* ]]; then
    echo "$speedup" >> results_row_cache/speedup.csv
    echo "Speedup extracted and saved: $speedup" >> "results_row_cache/intermediate_results/output_${size}.txt"
  else
    echo "Warning: Could not extract valid speedup for size $size. Raw output saved in results_row_cache/raw_output/output_${size}.txt"
    echo "Warning: Could not extract valid speedup" >> "results_row_cache/intermediate_results/output_${size}.txt"
  fi
  
  # 保存内存使用情况
  echo "Memory usage after test:" >> "results_row_cache/intermediate_results/output_${size}.txt"
  ps -o pid,rss,command | grep "gaussian_elimination_row_cache" | grep -v "grep" >> "results_row_cache/intermediate_results/output_${size}.txt" || echo "No process found" >> "results_row_cache/intermediate_results/output_${size}.txt"

  echo "Completed test for size $size"
  echo "========== End of Test for Matrix Size: $size ==========\n" >> "results_row_cache/intermediate_results/output_${size}.txt"
done

# 显示CSV文件内容以便调试
echo "Contents of execution_time.csv:"
cat results_row_cache/execution_time.csv
echo ""
echo "Contents of speedup.csv:"
cat results_row_cache/speedup.csv
echo ""

# 合并所有的中间结果到一个文件
echo "Combining all intermediate results into one file..."
cat results_row_cache/intermediate_results/output_*.txt > results_row_cache/output.txt
echo "Combined output saved to results_row_cache/output.txt"

echo "All tests completed. Results saved in results_row_cache directory."
echo "Raw outputs for each run saved in results_row_cache/raw_output directory."
echo "Intermediate results saved in results_row_cache/intermediate_results directory."
echo "Combined intermediate results saved in results_row_cache/output.txt."

# 使用Python绘制图表
echo "Generating plots..."
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
time_csv_path = 'results_row_cache/execution_time.csv'
speedup_csv_path = 'results_row_cache/speedup.csv'

# 显示文件内容
print('Contents of execution_time.csv:')
if os.path.exists(time_csv_path):
    with open(time_csv_path, 'r') as f:
        print(f.read())
else:
    print('File not found')

print('Contents of speedup.csv:')
if os.path.exists(speedup_csv_path):
    with open(speedup_csv_path, 'r') as f:
        print(f.read())
else:
    print('File not found')

try:
    # 检查文件是否只有表头行
    if os.path.exists(time_csv_path):
        with open(time_csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:
                print('Error: Execution time CSV file only contains the header row. No data was written.')
                with open('results_row_cache/plot_error.log', 'w') as error_file:
                    error_file.write('Execution time CSV file only contains the header row. No data was written.\n')
                exit(1)
    
    if os.path.exists(speedup_csv_path):
        with open(speedup_csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:
                print('Error: Speedup CSV file only contains the header row. No data was written.')
                with open('results_row_cache/plot_error.log', 'w') as error_file:
                    error_file.write('Speedup CSV file only contains the header row. No data was written.\n')
                exit(1)
    
    # 读取执行时间数据
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
    numeric_cols = time_data.columns.drop('matrix_size') if 'matrix_size' in time_data.columns else time_data.columns
    for col in numeric_cols:
        time_data[col] = pd.to_numeric(time_data[col], errors='coerce')
    
    numeric_cols = speedup_data.columns.drop('matrix_size') if 'matrix_size' in speedup_data.columns else speedup_data.columns
    for col in numeric_cols:
        speedup_data[col] = pd.to_numeric(speedup_data[col], errors='coerce')
    
    # 计算矩阵元素数量
    time_data['elements'] = time_data['matrix_size'].astype(int) * time_data['matrix_size'].astype(int)
    
    # 将结果保存到中间结果文件
    with open('results_row_cache/output.txt', 'a') as f:
        f.write('\n\n===== PROCESSING RESULTS =====\n')
        f.write('Execution Time Data:\n')
        f.write(str(time_data) + '\n\n')
        f.write('Speedup Data:\n')
        f.write(str(speedup_data) + '\n\n')
        f.write('Data with calculated elements count:\n')
        f.write(str(time_data) + '\n\n')
    
    # 执行时间图
    plt.figure(figsize=(12, 8))
    plt.plot(time_data['elements'], time_data['serial'] / 1000000, 'o-', label='Serial')
    plt.plot(time_data['elements'], time_data['serial_blocked'] / 1000000, '*-', label='Serial Blocked')
    plt.plot(time_data['elements'], time_data['dynamic_thread'] / 1000000, 's-', label='Dynamic Thread (Row)')
    plt.plot(time_data['elements'], time_data['static_semaphore'] / 1000000, '^-', label='Static Semaphore (Row)')
    plt.plot(time_data['elements'], time_data['static_full'] / 1000000, 'd-', label='Static Full (Row)')
    plt.plot(time_data['elements'], time_data['barrier'] / 1000000, 'x-', label='Barrier (Row)')
    plt.title('Cache-Optimized Row-Based Gaussian Elimination Execution Time on ARM Platform', fontsize=16)
    plt.xlabel('Matrix Elements Count', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_row_cache/execution_time_plot.png', dpi=300)
    print('Execution time plot generated.')
    
    # 加速比图
    plt.figure(figsize=(12, 8))
    plt.plot(speedup_data['matrix_size'], speedup_data['serial_blocked'], '*-', label='Serial Blocked')
    plt.plot(speedup_data['matrix_size'], speedup_data['dynamic_thread'], 's-', label='Dynamic Thread (Row)')
    plt.plot(speedup_data['matrix_size'], speedup_data['static_semaphore'], '^-', label='Static Semaphore (Row)')
    plt.plot(speedup_data['matrix_size'], speedup_data['static_full'], 'd-', label='Static Full (Row)')
    plt.plot(speedup_data['matrix_size'], speedup_data['barrier'], 'x-', label='Barrier (Row)')
    plt.title('Cache-Optimized Row-Based Gaussian Elimination Speedup on ARM Platform', fontsize=16)
    plt.xlabel('Matrix Size (rows)', fontsize=14)
    plt.ylabel('Speedup (compared to serial version)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results_row_cache/speedup_plot.png', dpi=300)
    print('Speedup plot generated.')

    # 与非缓存优化版本的对比图 (如果数据存在)
    try:
        # 尝试读取非缓存优化版本的数据
        standard_time_path = 'results_row/execution_time.csv'
        if os.path.exists(standard_time_path):
            standard_time_data = pd.read_csv(standard_time_path)
            numeric_cols = standard_time_data.columns.drop('matrix_size') if 'matrix_size' in standard_time_data.columns else standard_time_data.columns
            for col in numeric_cols:
                standard_time_data[col] = pd.to_numeric(standard_time_data[col], errors='coerce')
            standard_time_data['elements'] = standard_time_data['matrix_size'].astype(int) * standard_time_data['matrix_size'].astype(int)

            # 创建对比图
            plt.figure(figsize=(12, 8))
            
            # 标准版本
            plt.plot(standard_time_data['elements'], standard_time_data['serial'] / 1000000, 'o-', label='Standard Serial')
            plt.plot(standard_time_data['elements'], standard_time_data['dynamic_thread'] / 1000000, 's--', label='Standard Dynamic Thread')
            plt.plot(standard_time_data['elements'], standard_time_data['barrier'] / 1000000, 'x--', label='Standard Barrier')
            
            # 缓存优化版本
            plt.plot(time_data['elements'], time_data['serial'] / 1000000, 'o-', color='red', label='Cache-Optimized Serial')
            plt.plot(time_data['elements'], time_data['serial_blocked'] / 1000000, '*-', color='purple', label='Cache-Optimized Blocked')
            plt.plot(time_data['elements'], time_data['dynamic_thread'] / 1000000, 's-', color='green', label='Cache-Optimized Dynamic Thread')
            plt.plot(time_data['elements'], time_data['barrier'] / 1000000, 'x-', color='orange', label='Cache-Optimized Barrier')
            
            plt.title('Standard vs Cache-Optimized Row-Based Gaussian Elimination Comparison', fontsize=16)
            plt.xlabel('Matrix Elements Count', fontsize=14)
            plt.ylabel('Execution Time (seconds)', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig('results_row_cache/comparison_plot.png', dpi=300)
            print('Comparison plot generated.')
    except Exception as e:
        print(f'Warning: Could not generate comparison plot: {e}')
    
    # 记录到中间结果文件
    with open('results_row_cache/output.txt', 'a') as f:
        f.write('===== PLOTS GENERATED =====\n')
        f.write('Plot files:\n')
        f.write('- results_row_cache/execution_time_plot.png\n')
        f.write('- results_row_cache/speedup_plot.png\n')
        f.write('- results_row_cache/comparison_plot.png (if data available)\n')
        f.write('===== END OF PROCESSING =====\n')
    
    print('Plots generated successfully!')
    
except Exception as e:
    print(f'Error in Python script: {str(e)}')
    with open('results_row_cache/plot_error.log', 'w') as error_file:
        error_file.write(f'Error: {str(e)}\n')
    exit(1)
"

# 检查绘图是否成功
if [ $? -ne 0 ]; then
  echo "Failed to generate plots. Check results_row_cache/plot_error.log for details."
  echo "Make sure matplotlib and pandas are installed (pip3 install matplotlib pandas)."
  exit 1
fi
echo "Plots saved in results_row_cache directory."
echo "Done!"