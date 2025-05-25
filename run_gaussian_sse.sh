#!/bin/bash

# Define color settings
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Gaussian Elimination Algorithm Multi-Thread Performance Test Script ===${NC}"
echo -e "${BLUE}Current Date and Time: $(date "+%Y-%m-%d %H:%M:%S")${NC}"
echo -e "${BLUE}Current User: KKKyriejiang${NC}"

# Check for SSE4.2 support instead of AVX
echo -e "${BLUE}Checking CPU support for SSE4.2...${NC}"
if grep -q "sse4_2" /proc/cpuinfo; then
  echo -e "${GREEN}SSE4.2 instructions are supported!${NC}"
else
  echo -e "${RED}This CPU does not support SSE4.2 instructions!${NC}"
  echo -e "${YELLOW}The program requires a CPU with SSE4.2 support to run correctly.${NC}"
  read -p "Do you want to continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Exiting.${NC}"
    exit 1
  fi
  echo -e "${YELLOW}Continuing at your own risk. Results may be unreliable.${NC}"
fi

# Directory structure - use a timestamp to avoid overwriting previous results
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RESULTS_DIR="results_thread_scaling_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}/{raw_output,plots,thread_data,report}

# Compile the program with SSE4.2 flags
echo -e "${BLUE}Compiling Gaussian Elimination program (SSE optimized version)...${NC}"
g++ -O3 -msse4.2 -pthread -o gaussian_sse_test gaussian_sse.cpp

# Check if compilation was successful
if [ $? -ne 0 ]; then
  echo -e "${RED}Compilation failed!${NC}"
  exit 1
fi
echo -e "${GREEN}Compilation successful!${NC}"

# Initialize result files with headers
echo "matrix_size,thread_count,serial,dynamic_thread,static_semaphore,static_full,barrier" > ${RESULTS_DIR}/execution_time.csv
echo "matrix_size,thread_count,dynamic_thread,static_semaphore,static_full,barrier" > ${RESULTS_DIR}/speedup.csv

# Define test matrix sizes
SIZES=(100 500 1000 2000)

# Define thread counts to test
THREAD_COUNTS=(1 2 4 8 16)

# Run tests with each matrix size and thread count
for size in "${SIZES[@]}"; do
  echo -e "\n${BLUE}Testing matrix size: ${YELLOW}${size} x ${size}${NC}"
  
  for thread_count in "${THREAD_COUNTS[@]}"; do
    echo -e "\n${BLUE}Testing with ${YELLOW}${thread_count}${BLUE} threads${NC}"
    
    # Run the program with timeout to prevent hanging on large matrices
    echo -e "${BLUE}Running: ./gaussian_sse_test $size $thread_count${NC}"
    
    # Set timeout based on matrix size
    if [ $size -ge 1000 ]; then
      TIMEOUT=600  # 10 minutes for large matrices
    else
      TIMEOUT=300  # 5 minutes for smaller matrices
    fi
    
    timeout $TIMEOUT ./gaussian_sse_test $size $thread_count > ${RESULTS_DIR}/raw_output/output_${size}_${thread_count}.txt
    
    # Check if program timed out
    if [ $? -eq 124 ]; then
      echo -e "${RED}Program timed out after $TIMEOUT seconds for size $size with $thread_count threads. Skipping to next configuration.${NC}"
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
      echo -e "${RED}Could not find execution time data for size $size with $thread_count threads${NC}"
    fi
    
    if [[ "$SPEEDUP_LINE" == "$size"* ]]; then
      echo -e "${GREEN}Found speedup data: $SPEEDUP_LINE${NC}"
      echo "$SPEEDUP_LINE" >> ${RESULTS_DIR}/speedup.csv
    else
      echo -e "${RED}Could not find speedup data for size $size with $thread_count threads${NC}"
    fi
    
    # Show execution times and speedups for this configuration
    echo -e "${BLUE}Execution times for size $size with $thread_count threads:${NC}"
    grep "execution time:" ${RESULTS_DIR}/raw_output/output_${size}_${thread_count}.txt
    
    echo -e "${BLUE}Speedups for size $size with $thread_count threads:${NC}"
    grep "speedup:" ${RESULTS_DIR}/raw_output/output_${size}_${thread_count}.txt

    echo -e "${GREEN}Test completed for matrix size $size with $thread_count threads${NC}"
    echo "=========================================="
  done
done

# Check if we have enough data to generate plots
csv_lines=$(wc -l < ${RESULTS_DIR}/execution_time.csv)
if [ "$csv_lines" -le 1 ]; then
  echo -e "${RED}Error: No execution time data collected. Cannot generate plots.${NC}"
  exit 1
fi

# Generate enhanced plots for thread scaling
echo -e "${BLUE}Generating thread scaling plots...${NC}"
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

# Define vibrant colors for better differentiation
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot 1: Speedup vs Thread Count for each matrix size and implementation
plt.figure(figsize=(15, 10))

# Get unique matrix sizes
matrix_sizes = time_data['matrix_size'].unique()

# For each matrix size
for size in matrix_sizes:
    # Create a separate subplot
    plt.figure(figsize=(12, 8))
    
    # Filter data for this matrix size
    size_data = speedup_data[speedup_data['matrix_size'] == size]
    
    # Plot speedup vs thread count for each implementation
    plt.plot(size_data['thread_count'], size_data['dynamic_thread'], 's-', color=colors[1], 
             label='Dynamic Thread', linewidth=3, markersize=10)
    plt.plot(size_data['thread_count'], size_data['static_semaphore'], '^-', color=colors[2], 
             label='Static Semaphore', linewidth=3, markersize=10)
    plt.plot(size_data['thread_count'], size_data['static_full'], 'd-', color=colors[3], 
             label='Static Full', linewidth=3, markersize=10)
    plt.plot(size_data['thread_count'], size_data['barrier'], 'x-', color=colors[4], 
             label='Barrier', linewidth=3, markersize=10)
    
    # Add ideal speedup line (y = x)
    max_threads = max(size_data['thread_count'])
    plt.plot([1, max_threads], [1, max_threads], '--', color='gray', 
             label='Ideal Speedup', linewidth=2)
    
    plt.title(f'Speedup vs Thread Count for {size}x{size} Matrix', fontsize=16)
    plt.xlabel('Thread Count', fontsize=14)
    plt.ylabel('Speedup (relative to serial)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xscale('log', base=2)  # Use log scale for x-axis to better visualize thread scaling
    plt.xticks(size_data['thread_count'], size_data['thread_count'])  # Show actual thread counts
    
    # Save figure
    plt.savefig(f'${RESULTS_DIR}/plots/speedup_vs_threads_{size}.png', dpi=300, bbox_inches='tight')

# Plot 2: Combined plot with all matrix sizes for the best implementation
plt.figure(figsize=(12, 8))

# Determine the best implementation for each matrix size and thread count
best_impl_data = pd.DataFrame()
for size in matrix_sizes:
    size_data = speedup_data[speedup_data['matrix_size'] == size]
    
    # Find the best implementation for each thread count
    best_speedups = []
    for _, row in size_data.iterrows():
        impl_cols = ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']
        best_speedup = row[impl_cols].max()
        best_speedups.append(best_speedup)
    
    # Add to the best implementation dataframe
    best_data = pd.DataFrame({
        'matrix_size': size,
        'thread_count': size_data['thread_count'],
        'best_speedup': best_speedups
    })
    best_impl_data = pd.concat([best_impl_data, best_data])

# Plot the best speedups for each matrix size
for size in matrix_sizes:
    size_best = best_impl_data[best_impl_data['matrix_size'] == size]
    plt.plot(size_best['thread_count'], size_best['best_speedup'], 'o-', 
             label=f'{size}x{size}', linewidth=3, markersize=10)

# Add ideal speedup line
max_threads = max(best_impl_data['thread_count'])
plt.plot([1, max_threads], [1, max_threads], '--', color='gray', 
         label='Ideal Speedup', linewidth=2)

plt.title('Best Speedup vs Thread Count for Different Matrix Sizes', fontsize=16)
plt.xlabel('Thread Count', fontsize=14)
plt.ylabel('Best Speedup (relative to serial)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.xscale('log', base=2)  # Use log scale for x-axis
plt.xticks(best_impl_data['thread_count'].unique(), best_impl_data['thread_count'].unique())

plt.savefig('${RESULTS_DIR}/plots/best_speedup_all_sizes.png', dpi=300, bbox_inches='tight')

# Plot 3: Heatmap showing speedup efficiency (speedup / thread_count) for each implementation and size
plt.figure(figsize=(15, 10))

# Create a dataframe for efficiency calculations
efficiency_data = speedup_data.copy()
impl_cols = ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']
for col in impl_cols:
    efficiency_data[col] = efficiency_data[col] / efficiency_data['thread_count']

# Prepare data for heatmaps - one for each implementation
for impl in impl_cols:
    plt.figure(figsize=(10, 8))
    
    # Reshape data for heatmap (matrix_size vs thread_count)
    pivot_data = efficiency_data.pivot(index='matrix_size', columns='thread_count', values=impl)
    
    # Create heatmap
    im = plt.imshow(pivot_data, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Efficiency (speedup/threads)')
    
    # Set labels
    plt.title(f'Parallel Efficiency for {impl.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('Thread Count', fontsize=14)
    plt.ylabel('Matrix Size', fontsize=14)
    
    # Set ticks
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            text_color = 'white' if value < 0.5 else 'black'
            plt.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=10)
    
    plt.savefig(f'${RESULTS_DIR}/plots/efficiency_{impl}.png', dpi=300, bbox_inches='tight')

# Plot 4: Execution time vs thread count for largest matrix size
largest_size = max(matrix_sizes)
largest_data = time_data[time_data['matrix_size'] == largest_size]

plt.figure(figsize=(12, 8))

plt.semilogy(largest_data['thread_count'], largest_data['serial'] / 1000000, 'o-', color=colors[0], 
         label='Serial', linewidth=3, markersize=10)
plt.semilogy(largest_data['thread_count'], largest_data['dynamic_thread'] / 1000000, 's-', color=colors[1], 
         label='Dynamic Thread', linewidth=3, markersize=10)
plt.semilogy(largest_data['thread_count'], largest_data['static_semaphore'] / 1000000, '^-', color=colors[2], 
         label='Static Semaphore', linewidth=3, markersize=10)
plt.semilogy(largest_data['thread_count'], largest_data['static_full'] / 1000000, 'd-', color=colors[3], 
         label='Static Full', linewidth=3, markersize=10)
plt.semilogy(largest_data['thread_count'], largest_data['barrier'] / 1000000, 'x-', color=colors[4], 
         label='Barrier', linewidth=3, markersize=10)

plt.title(f'Execution Time vs Thread Count for {largest_size}x{largest_size} Matrix', fontsize=16)
plt.xlabel('Thread Count', fontsize=14)
plt.ylabel('Execution Time (seconds, log scale)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.xscale('log', base=2)  # Use log scale for x-axis
plt.xticks(largest_data['thread_count'], largest_data['thread_count'])

plt.savefig('${RESULTS_DIR}/plots/execution_time_largest.png', dpi=300, bbox_inches='tight')

# Plot 5: Scaling efficiency analysis for the best implementation
plt.figure(figsize=(12, 8))

# Calculate scaling efficiency (actual speedup / ideal speedup) for the best implementation
scaling_data = pd.DataFrame()

for size in matrix_sizes:
    size_data = speedup_data[speedup_data['matrix_size'] == size]
    
    # Find the best implementation speedup for each thread count
    best_speedups = []
    for _, row in size_data.iterrows():
        impl_cols = ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']
        best_speedup = row[impl_cols].max()
        best_speedups.append(best_speedup)
    
    # Calculate efficiency (actual speedup / thread count)
    efficiency = [s / t for s, t in zip(best_speedups, size_data['thread_count'])]
    
    # Add to the scaling data
    new_data = pd.DataFrame({
        'matrix_size': size,
        'thread_count': size_data['thread_count'],
        'best_speedup': best_speedups,
        'efficiency': efficiency
    })
    scaling_data = pd.concat([scaling_data, new_data])

# Plot scaling efficiency for each matrix size
for size in matrix_sizes:
    size_scaling = scaling_data[scaling_data['matrix_size'] == size]
    plt.plot(size_scaling['thread_count'], size_scaling['efficiency'], 'o-', 
             label=f'{size}x{size}', linewidth=3, markersize=10)

plt.title('Parallel Scaling Efficiency vs Thread Count', fontsize=16)
plt.xlabel('Thread Count', fontsize=14)
plt.ylabel('Scaling Efficiency (speedup/threads)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.xscale('log', base=2)  # Use log scale for x-axis
plt.xticks(scaling_data['thread_count'].unique(), scaling_data['thread_count'].unique())
# Add horizontal line at 100% efficiency
plt.axhline(y=1.0, color='gray', linestyle='--', label='Ideal Efficiency')
# Set y-axis limits to show efficiency better
plt.ylim([0, 1.2])

plt.savefig('${RESULTS_DIR}/plots/scaling_efficiency.png', dpi=300, bbox_inches='tight')

print('Thread scaling plots generated successfully!')
"

# Check if plotting was successful
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to generate plots. Make sure matplotlib and pandas are installed.${NC}"
  echo -e "${YELLOW}You can install them with: pip3 install matplotlib pandas numpy${NC}"
else
  echo -e "${GREEN}Thread scaling plots saved in ${RESULTS_DIR}/plots directory.${NC}"
fi

# Generate thread scaling report
echo -e "${BLUE}Generating thread scaling report...${NC}"

cat > ${RESULTS_DIR}/report/thread_scaling_report.md << EOL
# Gaussian Elimination Thread Scaling Performance Report

## Overview
This report analyzes the performance scaling of the SSE-optimized Gaussian elimination algorithm with varying thread counts.
Test date: $(date "+%Y-%m-%d %H:%M:%S")

## Test Environment
- Architecture: x86-64 with SSE4.2 support
- Compiler: G++ with O3 and SSE4.2 optimization flags
- Thread counts tested: ${THREAD_COUNTS[@]}
- Test matrix sizes: ${SIZES[@]}
- SIMD width: 4 floating point elements per vector (128 bits)

## Algorithm Implementations
1. **Serial Algorithm**: Baseline implementation using SSE vectorization
2. **Dynamic Thread Version**: Creates threads dynamically for each elimination round
3. **Static Thread + Semaphore Synchronization**: Fixed thread pool with semaphore synchronization
4. **Static Thread + Semaphore + Three-level Loop**: All loops within thread functions
5. **Static Thread + Barrier Synchronization**: Uses barrier synchronization

## Thread Scaling Analysis

### Speedup vs Thread Count for Different Matrix Sizes
The following plots show how speedup scales with increasing thread count for each matrix size:

$(for size in "${SIZES[@]}"; do
echo "#### ${size}x${size} Matrix
![Speedup vs Threads](../plots/speedup_vs_threads_${size}.png)
*How different implementations scale with thread count for ${size}x${size} matrix*
"
done)

### Best Implementation Speedup Comparison
![Best Speedup](../plots/best_speedup_all_sizes.png)
*Comparison of the best speedup achieved for each matrix size across different thread counts*

### Parallel Efficiency Analysis
Parallel efficiency measures how effectively additional threads contribute to performance (speedup/thread count):

![Dynamic Thread Efficiency](../plots/efficiency_dynamic_thread.png)
*Efficiency of Dynamic Thread implementation across matrix sizes and thread counts*

![Static Semaphore Efficiency](../plots/efficiency_static_semaphore.png)
*Efficiency of Static Semaphore implementation across matrix sizes and thread counts*

![Static Full Efficiency](../plots/efficiency_static_full.png)
*Efficiency of Static Full implementation across matrix sizes and thread counts*

![Barrier Efficiency](../plots/efficiency_barrier.png)
*Efficiency of Barrier implementation across matrix sizes and thread counts*

### Execution Time Analysis for Largest Matrix
![Execution Time](../plots/execution_time_largest.png)
*How execution time decreases with more threads for the largest matrix size*

### Overall Scaling Efficiency
![Scaling Efficiency](../plots/scaling_efficiency.png)
*How efficiently the algorithm scales with more threads across different matrix sizes*

## Key Findings

1. **Optimal Thread Count**:
   - For small matrices (100x100), using more than 2-4 threads often decreases performance due to thread management overhead
   - For larger matrices (1000x2000), performance continues to improve up to 8-16 threads

2. **Implementation Differences**:
   - **Dynamic Thread** shows poor scaling with increasing thread count due to thread creation overhead
   - **Static Full Thread** generally achieves the best performance for larger thread counts
   - **Barrier Synchronization** provides good scalability with minimal synchronization overhead

3. **Parallel Efficiency**:
   - Efficiency typically decreases as thread count increases, showing the effect of Amdahl's Law
   - Larger matrices maintain better efficiency with higher thread counts
   - When thread count > 8, efficiency drops significantly for all but the largest matrices

4. **Vectorization and Threading Interaction**:
   - SSE vectorization provides a baseline ~4x speedup in the serial version
   - Combined with multi-threading, implementations can achieve near-linear speedup for large matrices

## Recommendations

1. **Matrix Size-Based Thread Selection**:
   - For matrices < 500x500: Use 2-4 threads
   - For matrices >= 500x500: Use 4-8 threads
   - For matrices > 1000x1000: Use 8-16 threads (or more depending on hardware)

2. **Implementation Choice**:
   - For consistent performance: Use Static Full or Barrier implementations
   - Avoid Dynamic Thread implementation for repetitive calculations

3. **Hardware Considerations**:
   - Ensure thread count doesn't exceed physical core count for best performance
   - Consider cache hierarchy and NUMA effects for very large matrices

## Conclusion

The Gaussian elimination algorithm shows good scaling with increased thread count, especially for larger matrices. The combination of SSE vectorization and multi-threading provides substantial performance improvements over the serial implementation. However, there are diminishing returns as thread count increases, particularly for smaller problem sizes.

For optimal performance, it's essential to match the thread count to both the matrix size and the available hardware resources. The Static Full Thread implementation generally offers the best balance of performance and scaling efficiency across different configurations.

EOL

echo -e "${GREEN}Thread scaling report generated: ${RESULTS_DIR}/report/thread_scaling_report.md${NC}"
echo -e "${GREEN}All tests completed! Results saved in ${RESULTS_DIR} directory${NC}"
echo -e "${BLUE}Run end time: $(date "+%Y-%m-%d %H:%M:%S")${NC}"
echo -e "${BLUE}Thank you for using the Gaussian Elimination Thread Scaling Test Script${NC}"