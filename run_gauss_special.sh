#!/bin/bash

# 设置脚本环境
set -e  # 发生错误时退出
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# 颜色设置以提高可读性
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 特殊高斯消去算法测试脚本 (KKKyriejiang) - 快速测试版 v2 ===${NC}"
echo -e "${BLUE}当前日期和时间: $(date)${NC}"

# 编译程序（使用 ARM 架构交叉编译器）
echo -e "${BLUE}为ARM平台交叉编译特殊高斯消去程序...${NC}"
aarch64-linux-gnu-g++ -std=c++11 -o special_gaussian_elimination special_gaussian_elimination.cpp -lpthread -O3 

# 检查编译是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}编译失败!${NC}"
  exit 1
fi
echo -e "${GREEN}编译成功!${NC}"

# 创建结果目录
mkdir -p results_special
# 创建用于存放原始输出和中间结果的子目录
mkdir -p results_special/raw_output
mkdir -p results_special/intermediate_results
mkdir -p results_special/performance_data

# 测试参数设置
echo -e "${BLUE}准备运行小规模快速测试...${NC}"
# 矩阵规模参数 (被消元行数) - 修改为仅一个最小尺寸以加快测试
ROW_SIZES=(16) # <--- MODIFIED FOR SPEED
# 列数
COL_SIZE=1024 # (可以进一步减小以获得更快速度, 例如 256 或 512)
# 批次大小
BATCH_SIZE=64

# 清空并初始化结果文件
echo "rows,cols,batch_size,serial,dynamic_thread,static_semaphore,static_full,barrier" > results_special/execution_time.csv
echo "rows,cols,batch_size,dynamic_thread,static_semaphore,static_full,barrier" > results_special/speedup.csv

# 对每个矩阵大小运行测试（通过 QEMU 执行 ARM 可执行文件）
for rows in "${ROW_SIZES[@]}"; do
  echo -e "${BLUE}测试规模: ${YELLOW}${rows}行 x ${COL_SIZE}列, 批次大小:${BATCH_SIZE}${NC}"

  result_file="results_special/intermediate_results/output_${rows}.txt"
  echo "=== 特殊高斯消去测试 (${rows}行 x ${COL_SIZE}列, 批次大小:${BATCH_SIZE}) ===" > "$result_file"
  echo "命令: qemu-aarch64 -L /usr/aarch64-linux-gnu ./special_gaussian_elimination $rows $COL_SIZE $BATCH_SIZE 5 random" >> "$result_file"
  echo "开始时间: $(date)" >> "$result_file"
  echo "----------------------------------------" >> "$result_file"
  
  export QEMU_RESERVED_VA=8G
  export QEMU_HUGETLB=1
  
  echo -e "${BLUE}运行程序 (各种模式)...${NC}"
  output=$(qemu-aarch64 -L /usr/aarch64-linux-gnu ./special_gaussian_elimination $rows $COL_SIZE $BATCH_SIZE 5 random)
  
  echo -e "${GREEN}程序已完成执行. 结果摘要:${NC}"
  # Display more lines from output to help debug parsing
  echo "$output" | grep -E "执行时间|加速比|结果正确性" | head -15 

  echo "保存详细输出数据..."
  echo "$output" > "results_special/raw_output/output_${rows}.txt"
  echo "$output" >> "$result_file"
  echo "----------------------------------------" >> "$result_file"
  echo "结束时间: $(date)" >> "$result_file"
  
  # 提取执行时间数据
  serial_time=$(echo "$output" | grep "串行算法执行时间:" | sed 's/.*: \([0-9]*\) .*/\1/')
  dynamic_time=$(echo "$output" | grep "动态线程算法执行时间:" | sed 's/.*: \([0-9]*\) .*/\1/')
  semaphore_time=$(echo "$output" | grep "静态线程+信号量同步算法执行时间:" | sed 's/.*: \([0-9]*\) .*/\1/')
  full_time=$(echo "$output" | grep "静态线程+信号量同步+三重循环算法执行时间:" | sed 's/.*: \([0-9]*\) .*/\1/')
  barrier_time=$(echo "$output" | grep "静态线程+barrier同步算法执行时间:" | sed 's/.*: \([0-9]*\) .*/\1/')
  
  # 提取加速比数据 - MODIFIED -A 1 to -A 2
  dynamic_speedup=$(echo "$output" | grep -A 2 "动态线程算法执行时间:" | grep "加速比:" | sed 's/.*: \([0-9.]*\).*/\1/')
  semaphore_speedup=$(echo "$output" | grep -A 2 "静态线程+信号量同步算法执行时间:" | grep "加速比:" | sed 's/.*: \([0-9.]*\).*/\1/')
  full_speedup=$(echo "$output" | grep -A 2 "静态线程+信号量同步+三重循环算法执行时间:" | grep "加速比:" | sed 's/.*: \([0-9.]*\).*/\1/')
  barrier_speedup=$(echo "$output" | grep -A 2 "静态线程+barrier同步算法执行时间:" | grep "加速比:" | sed 's/.*: \([0-9.]*\).*/\1/')
  
  echo -e "执行时间数据: $serial_time, $dynamic_time, $semaphore_time, $full_time, $barrier_time"
  echo -e "加速比数据: $dynamic_speedup, $semaphore_speedup, $full_speedup, $barrier_speedup"

  if [ -n "$serial_time" ] && [ -n "$dynamic_time" ] && [ -n "$semaphore_time" ] && [ -n "$full_time" ] && [ -n "$barrier_time" ]; then
    echo "$rows,$COL_SIZE,$BATCH_SIZE,$serial_time,$dynamic_time,$semaphore_time,$full_time,$barrier_time" >> results_special/execution_time.csv
    echo "执行时间数据已保存" >> "$result_file"
  else
    echo -e "${YELLOW}警告: 无法提取有效的执行时间数据${NC}"
    echo "警告: 无法提取有效的执行时间数据" >> "$result_file"
  fi

  if [ -n "$dynamic_speedup" ] && [ -n "$semaphore_speedup" ] && [ -n "$full_speedup" ] && [ -n "$barrier_speedup" ]; then
    echo "$rows,$COL_SIZE,$BATCH_SIZE,$dynamic_speedup,$semaphore_speedup,$full_speedup,$barrier_speedup" >> results_special/speedup.csv
    echo "加速比数据已保存" >> "$result_file"
  else
    echo -e "${YELLOW}警告: 无法提取有效的加速比数据${NC}"
    echo "警告: 无法提取有效的加速比数据" >> "$result_file"
  fi
  
  echo "消元子数量估计: $(($COL_SIZE * 70 / 100))" >> "results_special/performance_data/stats_${rows}.txt"
  echo "被消元行数量: $rows" >> "results_special/performance_data/stats_${rows}.txt"
  echo "总列数: $COL_SIZE" >> "results_special/performance_data/stats_${rows}.txt"
  echo "批次大小: $BATCH_SIZE" >> "results_special/performance_data/stats_${rows}.txt"
  echo "填充率估计: $(echo "scale=2; $rows * $COL_SIZE * 20 / 100000" | bc)%" >> "results_special/performance_data/stats_${rows}.txt"
  
  echo -e "${BLUE}记录内存使用情况...${NC}"
  echo "内存使用统计:" >> "$result_file"
  ps -o pid,rss,command | grep "qemu-aarch64.*special_gaussian_elimination" | grep -v "grep" >> "$result_file" || echo "未找到进程" >> "$result_file"

  echo -e "${GREEN}完成规模为 $rows 的测试${NC}"
  echo "========== 测试结束 - 规模: $rows ==========\n" >> "$result_file"
done

echo -e "${BLUE}结果总结:${NC}"
echo "执行时间数据 (execution_time.csv):"
cat results_special/execution_time.csv
echo ""
echo "加速比数据 (speedup.csv):"
cat results_special/speedup.csv
echo ""

echo -e "${BLUE}合并所有中间结果到单个文件...${NC}"
cat results_special/intermediate_results/output_*.txt > results_special/output.txt
echo -e "${GREEN}合并输出已保存到 results_special/output.txt${NC}"

echo -e "${BLUE}生成图表...${NC}"
python3 -c "
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print('Python版本:', sys.version)
print('工作目录:', os.getcwd())

time_csv_path = 'results_special/execution_time.csv'
speedup_csv_path = 'results_special/speedup.csv'

try:
    print('读取CSV文件...')
    time_data = pd.read_csv(time_csv_path)
    speedup_data = pd.read_csv(speedup_csv_path)
    
    if time_data.empty or len(time_data) < 1:
        print(f'警告: {time_csv_path} 为空或数据点不足. 跳过相关绘图.')
    else:
        print('执行时间数据:')
        print(time_data)
        numeric_cols_time = time_data.columns.drop(['rows', 'cols', 'batch_size'])
        for col in numeric_cols_time:
            time_data[col] = pd.to_numeric(time_data[col], errors='coerce')
        time_data['total_elements'] = time_data['rows'] * time_data['cols']
        
        plt.figure(figsize=(12, 8))
        plt.plot(time_data['rows'], time_data['serial'] / 1000000, 'o-', label='串行算法', linewidth=2)
        plt.plot(time_data['rows'], time_data['dynamic_thread'] / 1000000, 's-', label='动态线程版本', linewidth=2)
        plt.plot(time_data['rows'], time_data['static_semaphore'] / 1000000, '^-', label='静态线程+信号量', linewidth=2)
        plt.plot(time_data['rows'], time_data['static_full'] / 1000000, 'd-', label='静态线程+信号量+三重循环', linewidth=2)
        plt.plot(time_data['rows'], time_data['barrier'] / 1000000, 'x-', label='静态线程+barrier', linewidth=2)
        plt.title('特殊高斯消去算法在ARM平台上的执行时间对比', fontsize=16)
        plt.xlabel('被消元行数量', fontsize=14)
        plt.ylabel('执行时间（秒）', fontsize=14)
        plt.grid(True); plt.legend(fontsize=12, loc='best'); plt.tight_layout()
        plt.savefig('results_special/execution_time_plot.png', dpi=300)
        print('执行时间图已生成.')

        if not time_data[['dynamic_thread', 'static_semaphore', 'static_full', 'barrier', 'serial']].isnull().all().all():
            plt.figure(figsize=(12, 8))
            plt.plot(time_data['rows'], time_data['static_full'] / 1000000, 'd-', label='静态线程+信号量+三重循环', linewidth=2)
            plt.plot(time_data['rows'], time_data['barrier'] / 1000000, 'x-', label='静态线程+barrier', linewidth=2)
            plt.title('最优两种并行策略执行时间对比', fontsize=16)
            plt.xlabel('被消元行数量', fontsize=14); plt.ylabel('执行时间（秒）', fontsize=14)
            plt.grid(True); plt.legend(fontsize=12, loc='best'); plt.tight_layout()
            plt.savefig('results_special/best_algorithms_comparison.png', dpi=300)
            print('最优算法对比图已生成.')

            plt.figure(figsize=(14, 10))
            for col in ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']:
                if col in time_data and 'serial' in time_data:
                     time_data[f'norm_{col}'] = time_data[col] / time_data['serial']
            
            x = np.arange(len(time_data['rows']))
            width = 0.15
            if f'norm_dynamic_thread' in time_data: plt.bar(x - width*2, time_data['norm_dynamic_thread'], width, label='动态线程版本')
            if f'norm_static_semaphore' in time_data: plt.bar(x - width, time_data['norm_static_semaphore'], width, label='静态线程+信号量')
            if f'norm_static_full' in time_data: plt.bar(x, time_data['norm_static_full'], width, label='静态线程+信号量+三重循环')
            if f'norm_barrier' in time_data: plt.bar(x + width, time_data['norm_barrier'], width, label='静态线程+barrier')
            plt.bar(x + width*2, np.ones(len(time_data['rows'])), width, label='串行算法（基准）')
            plt.xlabel('被消元行数量', fontsize=14); plt.ylabel('归一化执行时间（相对于串行算法）', fontsize=14)
            plt.title('特殊高斯消去算法各种实现的归一化执行时间', fontsize=16)
            plt.xticks(x, time_data['rows']); plt.legend(); plt.grid(True, axis='y'); plt.tight_layout()
            plt.savefig('results_special/normalized_execution_time.png', dpi=300)
            print('归一化执行时间图已生成.')

    if speedup_data.empty or len(speedup_data) < 1:
        print(f'警告: {speedup_csv_path} 为空或数据点不足. 跳过相关绘图.')
    else:
        print('加速比数据:')
        print(speedup_data)
        numeric_cols_speedup = speedup_data.columns.drop(['rows', 'cols', 'batch_size'])
        for col in numeric_cols_speedup:
            speedup_data[col] = pd.to_numeric(speedup_data[col], errors='coerce')

        plt.figure(figsize=(12, 8))
        plt.plot(speedup_data['rows'], speedup_data['dynamic_thread'], 's-', label='动态线程版本', linewidth=2)
        plt.plot(speedup_data['rows'], speedup_data['static_semaphore'], '^-', label='静态线程+信号量', linewidth=2)
        plt.plot(speedup_data['rows'], speedup_data['static_full'], 'd-', label='静态线程+信号量+三重循环', linewidth=2)
        plt.plot(speedup_data['rows'], speedup_data['barrier'], 'x-', label='静态线程+barrier', linewidth=2)
        plt.title('特殊高斯消去算法在ARM平台上的加速比对比', fontsize=16)
        plt.xlabel('被消元行数量', fontsize=14); plt.ylabel('加速比（相对于串行版本）', fontsize=14)
        plt.grid(True); plt.legend(fontsize=12, loc='best'); plt.tight_layout()
        plt.savefig('results_special/speedup_plot.png', dpi=300)
        print('加速比图已生成.')
        
    print('图表生成尝试完成.')
    
except FileNotFoundError:
    print(f'Python脚本错误: CSV文件未找到. Time: {time_csv_path}, Speedup: {speedup_csv_path}')
    with open('results_special/plot_error.log', 'w') as error_file: error_file.write(f'错误: CSV文件未找到.\n')
except Exception as e:
    print(f'Python脚本错误: {str(e)}')
    with open('results_special/plot_error.log', 'w') as error_file: error_file.write(f'错误: {str(e)}\n')
"

echo -e "${GREEN}图表生成尝试完成 (错误将被记录到 results_special/plot_error.log).${NC}"

echo -e "${BLUE}生成性能报告...${NC}"
cat > results_special/performance_report.md << EOL
# 特殊高斯消去算法性能报告 (快速测试版 v2)

## 概述
本报告总结了特殊高斯消去算法在ARM平台上的性能测试结果。
测试日期: $(date)
**重要: 当前并行算法版本存在正确性问题 (结果验证失败)，性能数据仅供参考，需首先解决正确性问题。**

## 测试环境
- 架构: ARM (通过QEMU模拟)
- 编译器: aarch64-linux-gnu-g++ 带O3优化
- 测试规模: ${ROW_SIZES[@]} (被消元行) × $COL_SIZE (列)
- 批次大小: $BATCH_SIZE

## 算法实现
1. **串行算法**: 基准实现
2. **动态线程版本**: (结果验证失败)
3. **静态线程+信号量同步版本**: (结果验证失败)
4. **静态线程+信号量同步+三重循环**: (结果验证失败)
5. **静态线程+barrier同步版本**: (结果验证失败 - 待确认)

## 性能总结 (基于快速测试数据 - **结果不正确**)

*(图表可能无法在此快速测试中正确生成或数据点不足)*

![执行时间](execution_time_plot.png)
*不同实现方式的执行时间对比*

![加速比](speedup_plot.png)
*相对于串行实现的加速比*

## 结论 (初步 - **以正确性为重**)
目前所有并行算法均未能通过正确性验证。在解决这些正确性问题之前，任何性能分析都是次要的。
建议集中精力调试 C++ 代码中的并行逻辑。

EOL

echo -e "${GREEN}性能报告已生成: results_special/performance_report.md${NC}"
echo -e "${GREEN}所有测试完成! 结果保存在 results_special 目录${NC}"
echo -e "${BLUE}运行结束时间: $(date)${NC}"
echo -e "${BLUE}感谢使用特殊高斯消去算法测试脚本 - by KKKyriejiang${NC}"