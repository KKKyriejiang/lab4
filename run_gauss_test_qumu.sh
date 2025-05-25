#!/bin/bash

# 编译程序（使用 ARM 架构交叉编译器）
echo "Cross-compiling Gaussian Elimination program for ARM..."
aarch64-linux-gnu-g++ -static -o gauss_elimination_arm gauss_elimination_arm.cpp -lpthread -O3

# 检查编译是否成功
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

# 创建结果目录
mkdir -p results
# 创建用于存放原始输出的子目录
mkdir -p results/raw_output

# 运行不同规模的矩阵测试
echo "Running tests with different matrix sizes..."
SIZES=(16 32 64 128)


# 清空结果文件
echo "matrix_size,serial,dynamic_thread,static_semaphore,static_full,barrier" > results/execution_time.csv
echo "matrix_size,dynamic_thread,static_semaphore,static_full,barrier" > results/speedup.csv

# 对每个矩阵大小运行测试（通过 QEMU 执行 ARM 可执行文件）
for size in "${SIZES[@]}"; do
  echo "Testing matrix size: $size"

  # 运行程序并提取结果
  output=$(qemu-aarch64 ./gauss_elimination_arm $size)

  # 保存当前规模的完整输出
  echo "Saving raw output for size $size..."
  echo "$output" > "results/raw_output/output_${size}.txt" # 新增行：保存原始输出

  # 提取CSV格式的数据
  execution_time=$(echo "$output" | grep -A 1 "CSV Format for plotting:" | tail -n 1)
  speedup=$(echo "$output" | grep -A 1 "Speedup CSV Format for plotting:" | tail -n 1)

  # 添加到结果文件
  # 检查提取的数据是否为空，避免写入空行到CSV
  if [ -n "$execution_time" ]; then
    echo "$execution_time" >> results/execution_time.csv
  else
    echo "Warning: Could not extract execution time for size $size. Raw output saved in results/raw_output/output_${size}.txt"
  fi

  if [ -n "$speedup" ]; then
    echo "$speedup" >> results/speedup.csv
  else
    echo "Warning: Could not extract speedup for size $size. Raw output saved in results/raw_output/output_${size}.txt"
  fi

  echo "Completed test for size $size"
done
echo "All tests completed. Results saved in results directory."
echo "Raw outputs for each run saved in results/raw_output directory."

# 使用Python绘制图表
echo "Generating plots..."
python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 检查结果文件是否存在且不为空
time_csv_path = 'results/execution_time.csv'
speedup_csv_path = 'results/speedup.csv'

if not os.path.exists(time_csv_path) or os.path.getsize(time_csv_path) <= $(wc -l < $time_csv_path) : # 检查表头行是否是唯一的行
    print(f'Error: {time_csv_path} is empty or only contains headers. Cannot generate plots.')
    exit(1)
if not os.path.exists(speedup_csv_path) or os.path.getsize(speedup_csv_path) <= $(wc -l < $speedup_csv_path) : # 检查表头行是否是唯一的行
    print(f'Error: {speedup_csv_path} is empty or only contains headers. Cannot generate plots.')
    exit(1)

# 读取执行时间数据
try:
    time_data = pd.read_csv(time_csv_path)
    speedup_data = pd.read_csv(speedup_csv_path)
except pd.errors.EmptyDataError:
    print('Error: One of the CSV files is empty. Cannot generate plots.')
    exit(1)

if time_data.empty or speedup_data.empty:
    print('Error: Dataframes are empty after reading CSV. Cannot generate plots.')
    exit(1)

# 计算矩阵元素数量（矩阵大小的平方）
time_data['elements'] = time_data['matrix_size'] * time_data['matrix_size']

# 执行时间 vs 矩阵元素数量图
plt.figure(figsize=(12, 8))
plt.plot(time_data['elements'], time_data['serial'] / 1000000, 'o-', label='Serial')
plt.plot(time_data['elements'], time_data['dynamic_thread'] / 1000000, 's-', label='Dynamic Thread')
plt.plot(time_data['elements'], time_data['static_semaphore'] / 1000000, '^-', label='Static Semaphore')
plt.plot(time_data['elements'], time_data['static_full'] / 1000000, 'd-', label='Static Full')
plt.plot(time_data['elements'], time_data['barrier'] / 1000000, 'x-', label='Barrier')
plt.title('Gaussian Elimination Execution Time on ARM Platform', fontsize=16)
plt.xlabel('Matrix Elements Count', fontsize=14)
plt.ylabel('Execution Time (seconds)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('results/execution_time_plot.png', dpi=300)
print('Execution time plot generated.')

# 加速比 vs 矩阵行数图
plt.figure(figsize=(12, 8))
plt.plot(speedup_data['matrix_size'], speedup_data['dynamic_thread'], 's-', label='Dynamic Thread')
plt.plot(speedup_data['matrix_size'], speedup_data['static_semaphore'], '^-', label='Static Semaphore')
plt.plot(speedup_data['matrix_size'], speedup_data['static_full'], 'd-', label='Static Full')
plt.plot(speedup_data['matrix_size'], speedup_data['barrier'], 'x-', label='Barrier')
plt.title('Gaussian Elimination Speedup on ARM Platform', fontsize=16)
plt.xlabel('Matrix Size (rows)', fontsize=14)
plt.ylabel('Speedup (compared to serial version)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('results/speedup_plot.png', dpi=300)
print('Speedup plot generated.')

print('Plots generated successfully!')
"

# 检查绘图是否成功
if [ $? -ne 0 ]; then
  echo "Failed to generate plots. Make sure matplotlib and pandas are installed."
  echo "You can install them using: pip3 install matplotlib pandas"
  exit 1
fi
echo "Plots saved in results directory."
echo "Done!"
