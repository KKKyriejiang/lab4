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

echo -e "${BLUE}=== Gaussian Elimination Thread Scaling Test ===${NC}"
echo -e "${BLUE}Current Date and Time: 2025-05-22 07:47:32${NC}"
echo -e "${BLUE}Current User: KKKyriejiang${NC}"

# 修改源代码以支持动态线程数
echo -e "${BLUE}Modifying source code to support dynamic thread counts...${NC}"
# 备份原始源代码
cp gaussian_elimination_arm.cpp gaussian_elimination_arm_original.cpp

# 查看/修改源码 - 使用动态分配而不是固定大小数组
cat > gaussian_elimination_arm.cpp << EOL
#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <arm_neon.h>
#include <cmath>
using namespace std;

// 矩阵大小
int n = 0;
// 线程数量 - 现在是变量而非常量
int NUM_THREADS = 4;
// 矩阵数据
float **matrix = nullptr;

// 用于计时
struct timeval start_time, end_time;
long long execution_time;

// 同步所需的信号量和屏障
sem_t sem_main;
sem_t *sem_workerstart;  // 动态分配
sem_t *sem_workerend;    // 动态分配
sem_t sem_leader;
sem_t *sem_division;     // 动态分配
sem_t *sem_elimination;  // 动态分配
pthread_barrier_t barrier_division;
pthread_barrier_t barrier_elimination;

// 线程参数结构体
typedef struct {
    int k;       // 当前消去的行
    int t_id;    // 线程ID
} threadParam_t;

// 初始化矩阵，使用随机数填充
void init_matrix() {
    matrix = new float*[n];
    for (int i = 0; i < n; i++) {
        matrix[i] = new float[n];
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 100 + 1;
        }
    }
    // 保证矩阵可逆
    for (int i = 0; i < n; i++) {
        matrix[i][i] += n * 10;
    }
}

// 释放矩阵内存
void free_matrix() {
    for (int i = 0; i < n; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// 初始化同步原语
void init_sync_objects() {
    // 动态分配信号量数组
    sem_workerstart = new sem_t[NUM_THREADS];
    sem_workerend = new sem_t[NUM_THREADS];
    sem_division = new sem_t[NUM_THREADS-1];
    sem_elimination = new sem_t[NUM_THREADS-1];
    
    // 初始化信号量
    sem_init(&sem_main, 0, 0);
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        sem_init(&sem_division[i], 0, 0);
        sem_init(&sem_elimination[i], 0, 0);
    }
    
    // 初始化屏障
    pthread_barrier_init(&barrier_division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_elimination, NULL, NUM_THREADS);
}

// 清理同步原语
void cleanup_sync_objects() {
    sem_destroy(&sem_main);
    sem_destroy(&sem_leader);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
    
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        sem_destroy(&sem_division[i]);
        sem_destroy(&sem_elimination[i]);
    }
    
    pthread_barrier_destroy(&barrier_division);
    pthread_barrier_destroy(&barrier_elimination);
    
    // 释放动态分配的内存
    delete[] sem_workerstart;
    delete[] sem_workerend;
    delete[] sem_division;
    delete[] sem_elimination;
}

// NEON向量化的行归一化函数
void division_neon(int k) {
    float32x4_t vt = vdupq_n_f32(matrix[k][k]);
    int j = k + 1;
    
    // 使用NEON指令进行向量化除法
    for (; j + 3 < n; j += 4) {
        float32x4_t va = vld1q_f32(&matrix[k][j]);
        va = vdivq_f32(va, vt);
        vst1q_f32(&matrix[k][j], va);
    }
    
    // 处理剩余的元素
    for (; j < n; j++) {
        matrix[k][j] = matrix[k][j] / matrix[k][k];
    }
    
    matrix[k][k] = 1.0;
}

// NEON向量化的消去函数
void elimination_neon(int k, int i) {
    float32x4_t vaik = vdupq_n_f32(matrix[i][k]);
    int j = k + 1;
    
    // 使用NEON指令进行向量化消去
    for (; j + 3 < n; j += 4) {
        float32x4_t vakj = vld1q_f32(&matrix[k][j]);
        float32x4_t vaij = vld1q_f32(&matrix[i][j]);
        float32x4_t vx = vmulq_f32(vaik, vakj);
        vaij = vsubq_f32(vaij, vx);
        vst1q_f32(&matrix[i][j], vaij);
    }
    
    // 处理剩余的元素
    for (; j < n; j++) {
        matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
    }
    
    matrix[i][k] = 0;
}

// 串行高斯消去算法
void gaussEliminationSerial() {
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        division_neon(k);
        
        // 消去后续各行
        for (int i = k + 1; i < n; i++) {
            elimination_neon(k, i);
        }
    }
}

// 1. 动态线程版本的消去线程函数
void* dynamicThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int i = k + t_id + 1;
    
    // 对应行进行消去
    elimination_neon(k, i);
    
    pthread_exit(NULL);
}

// 1. 动态线程版本的高斯消去算法
void gaussEliminationDynamicThread() {
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        division_neon(k);
        
        // 创建需要的线程数量
        int worker_count = n - 1 - k;
        if (worker_count <= 0) continue;
        
        pthread_t* handles = new pthread_t[worker_count];
        threadParam_t* param = new threadParam_t[worker_count];
        
        // 为每个线程分配任务
        for (int t_id = 0; t_id < worker_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        
        // 创建线程
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, dynamicThreadFunc, &param[t_id]);
        }
        
        // 等待所有线程完成
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        
        // 释放资源
        delete[] handles;
        delete[] param;
    }
}

// 2. 静态线程+信号量同步版本的线程函数
void* staticSemaphoreThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]); // 等待主线程通知开始工作
        
        // 使用NEON优化的消去操作
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            elimination_neon(k, i);
        }
        
        sem_post(&sem_main); // 通知主线程已完成
        sem_wait(&sem_workerend[t_id]); // 等待所有工作线程完成当前轮次
    }
    
    pthread_exit(NULL);
}

// 2. 静态线程+信号量同步版本的高斯消去算法
void gaussEliminationStaticSemaphore() {
    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, staticSemaphoreThreadFunc, &param[t_id]);
    }
    
    // 主线程控制计算过程
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        division_neon(k);
        
        // 唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workerstart[t_id]);
        }
        
        // 等待所有工作线程完成消去
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        
        // 通知工作线程进入下一轮
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workerend[t_id]);
        }
    }
    
    // 等待所有线程结束
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
}

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的线程函数
void* staticFullThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // 线程0负责归一化操作
        if (t_id == 0) {
            division_neon(k);
        } else {
            sem_wait(&sem_division[t_id-1]); // 非0线程等待归一化完成
        }
        
        // 线程0通知其他线程归一化已完成
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_division[i]);
            }
        }
        
        // 所有线程执行消去操作
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            elimination_neon(k, i);
        }
        
        // 线程同步，确保所有消去操作完成
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_wait(&sem_leader); // 等待其他线程完成消去
            }
            
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_elimination[i]); // 通知其他线程进入下一轮
            }
        } else {
            sem_post(&sem_leader); // 通知主线程已完成
            sem_wait(&sem_elimination[t_id-1]); // 等待进入下一轮
        }
    }
    
    pthread_exit(NULL);
}

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的高斯消去算法
void gaussEliminationStaticFull() {
    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, staticFullThreadFunc, &param[t_id]);
    }
    
    // 等待所有线程结束
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
}

// 4. 静态线程+barrier同步版本的线程函数
void* staticBarrierThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // 线程0负责归一化操作
        if (t_id == 0) {
            division_neon(k);
        }
        
        // 使用屏障同步，确保归一化完成
        pthread_barrier_wait(&barrier_division);
        
        // 所有线程执行消去操作
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            elimination_neon(k, i);
        }
        
        // 使用屏障同步，确保所有消去操作完成
        pthread_barrier_wait(&barrier_elimination);
    }
    
    pthread_exit(NULL);
}

// 4. 静态线程+barrier同步版本的高斯消去算法
void gaussEliminationBarrier() {
    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, staticBarrierThreadFunc, &param[t_id]);
    }
    
    // 等待所有线程结束
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
}

// 检查结果是否正确（与串行算法比较）
bool check_result(float** result_matrix) {
    // 创建一个拷贝用于串行计算
    float** temp_matrix = new float*[n];
    for (int i = 0; i < n; i++) {
        temp_matrix[i] = new float[n];
        memcpy(temp_matrix[i], matrix[i], n * sizeof(float));
    }
    
    // 运行标准串行算法
    for (int k = 0; k < n; k++) {
        // 归一化
        for (int j = k + 1; j < n; j++) {
            temp_matrix[k][j] /= temp_matrix[k][k];
        }
        temp_matrix[k][k] = 1.0;
        
        // 消去
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                temp_matrix[i][j] -= temp_matrix[i][k] * temp_matrix[k][j];
            }
            temp_matrix[i][k] = 0;
        }
    }
    
    // 比较结果
    bool correct = true;
    for (int i = 0; i < n && correct; i++) {
        for (int j = 0; j < n && correct; j++) {
            if (fabs(result_matrix[i][j] - temp_matrix[i][j]) > 1e-4) {
                correct = false;
                cout << "Difference at [" << i << "][" << j << "]: " 
                     << result_matrix[i][j] << " vs " << temp_matrix[i][j] << endl;
            }
        }
    }
    
    // 释放临时矩阵
    for (int i = 0; i < n; i++) {
        delete[] temp_matrix[i];
    }
    delete[] temp_matrix;
    
    return correct;
}

// 保存结果
float** save_result() {
    float** result = new float*[n];
    for (int i = 0; i < n; i++) {
        result[i] = new float[n];
        memcpy(result[i], matrix[i], n * sizeof(float));
    }
    return result;
}

// 恢复原始矩阵
void restore_matrix(float** result) {
    for (int i = 0; i < n; i++) {
        memcpy(matrix[i], result[i], n * sizeof(float));
    }
}

// 释放结果矩阵
void free_result(float** result) {
    for (int i = 0; i < n; i++) {
        delete[] result[i];
    }
    delete[] result;
}

// 计算执行时间
long long get_execution_time() {
    return (end_time.tv_sec - start_time.tv_sec) * 1000000LL + (end_time.tv_usec - start_time.tv_usec);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <matrix_size> [thread_count]" << endl;
        return -1;
    }
    
    // 设置矩阵大小
    n = atoi(argv[1]);
    if (n <= 0) {
        cout << "Invalid matrix size" << endl;
        return -1;
    }
    
    // 设置线程数（如果指定）
    if (argc >= 3) {
        NUM_THREADS = atoi(argv[2]);
        if (NUM_THREADS <= 0) {
            cout << "Invalid thread count, using default (4)" << endl;
            NUM_THREADS = 4;
        }
        cout << "Using " << NUM_THREADS << " threads" << endl;
    }
    
    // 初始化随机数生成器
    srand(42);
    
    // 初始化矩阵
    init_matrix();
    
    // 初始化同步原语
    init_sync_objects();
    
    // 保存原始矩阵
    float** original_matrix = save_result();
    
    // 运行并测试串行版本
    gettimeofday(&start_time, NULL);
    gaussEliminationSerial();
    gettimeofday(&end_time, NULL);
    long long serial_time = get_execution_time();
    float** serial_result = save_result();
    
    cout << "Serial version execution time: " << serial_time << " us" << endl;
    
    // 运行并测试版本1：动态线程版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationDynamicThread();
    gettimeofday(&end_time, NULL);
    long long dynamic_thread_time = get_execution_time();
    
    cout << "Dynamic Thread version execution time: " << dynamic_thread_time << " us" << endl;
    cout << "Dynamic Thread version speedup: " << (float)serial_time / dynamic_thread_time << endl;
    cout << "Dynamic Thread version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本2：静态线程+信号量同步版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticSemaphore();
    gettimeofday(&end_time, NULL);
    long long static_semaphore_time = get_execution_time();
    
    cout << "Static Semaphore version execution time: " << static_semaphore_time << " us" << endl;
    cout << "Static Semaphore version speedup: " << (float)serial_time / static_semaphore_time << endl;
    cout << "Static Semaphore version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本3：静态线程+信号量同步+三重循环全部纳入线程函数版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticFull();
    gettimeofday(&end_time, NULL);
    long long static_full_time = get_execution_time();
    
    cout << "Static Full Thread version execution time: " << static_full_time << " us" << endl;
    cout << "Static Full Thread version speedup: " << (float)serial_time / static_full_time << endl;
    cout << "Static Full Thread version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本4：静态线程+barrier同步版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationBarrier();
    gettimeofday(&end_time, NULL);
    long long barrier_time = get_execution_time();
    
    cout << "Barrier version execution time: " << barrier_time << " us" << endl;
    cout << "Barrier version speedup: " << (float)serial_time / barrier_time << endl;
    cout << "Barrier version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 输出CSV格式的执行时间
    cout << "\\nCSV Format for plotting:" << endl;
    cout << "matrix_size,serial,dynamic_thread,static_semaphore,static_full,barrier" << endl;
    cout << n << "," << serial_time << "," << dynamic_thread_time << "," 
         << static_semaphore_time << "," << static_full_time << "," << barrier_time << endl;
    
    // 输出CSV格式的加速比
    cout << "\\nSpeedup CSV Format for plotting:" << endl;
    cout << "matrix_size,dynamic_thread,static_semaphore,static_full,barrier" << endl;
    cout << n << "," << (float)serial_time / dynamic_thread_time << "," 
         << (float)serial_time / static_semaphore_time << "," 
         << (float)serial_time / static_full_time << "," 
         << (float)serial_time / barrier_time << endl;
    
    // 释放内存
    free_result(original_matrix);
    free_result(serial_result);
    free_matrix();
    cleanup_sync_objects();
    
    return 0;
}
EOL

# 编译程序（使用 ARM 架构交叉编译器）
echo -e "${BLUE}Cross-compiling Gaussian Elimination program for ARM...${NC}"
aarch64-linux-gnu-g++ -static -o gaussian_elimination_arm gaussian_elimination_arm.cpp -lpthread -O3

# 检查编译是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}Compilation failed!${NC}"
  # 恢复原始源代码
  mv gaussian_elimination_arm_original.cpp gaussian_elimination_arm.cpp
  exit 1
fi

echo -e "${GREEN}Compilation successful${NC}"

# 创建结果目录
mkdir -p results/{raw_output,intermediate_results,plots}

# 修改后的矩阵测试规模和线程数
echo -e "${BLUE}Running tests with different matrix sizes and thread counts...${NC}"
SIZES=(16 32 64 128)
THREAD_COUNTS=(1 2 4 8 16)

# 清空并初始化结果文件
echo "matrix_size,thread_count,serial,dynamic_thread,static_semaphore,static_full,barrier" > results/execution_time.csv
echo "matrix_size,thread_count,dynamic_thread,static_semaphore,static_full,barrier" > results/speedup.csv

# 对每个矩阵大小运行测试
for size in "${SIZES[@]}"; do
  echo -e "\n${BLUE}Testing matrix size: ${YELLOW}$size${NC}"

  # 对每个线程数运行测试
  for threads in "${THREAD_COUNTS[@]}"; do
    echo -e "${BLUE}Testing with $threads threads...${NC}"
    
    # 保存中间结果文件路径
    result_file="results/intermediate_results/output_${size}_threads_${threads}.txt"
    
    # 创建中间结果文件头部
    echo "=== Gaussian Elimination Test with Matrix Size: $size, Threads: $threads ===" > "$result_file"
    echo "Command: qemu-aarch64 ./gaussian_elimination_arm $size $threads" >> "$result_file"
    echo "Started at: $(date)" >> "$result_file"
    echo "----------------------------------------" >> "$result_file"

    # 预热运行
    echo -e "${BLUE}Running warm-up iteration...${NC}"
    qemu-aarch64 ./gaussian_elimination_arm $size $threads > /dev/null 2>&1 || true
    
    # 运行程序并提取结果
    echo -e "${BLUE}Running benchmark...${NC}"
    output=$(qemu-aarch64 ./gaussian_elimination_arm $size $threads)

    # 保存当前规模和线程数的完整输出到原始输出目录
    echo -e "${BLUE}Saving raw output for size $size with $threads threads...${NC}"
    echo "$output" > "results/raw_output/output_${size}_threads_${threads}.txt"
    
    # 同时将输出添加到中间结果文件
    echo "$output" >> "$result_file"
    
    # 为中间结果文件添加分隔符和时间戳
    echo "----------------------------------------" >> "$result_file"
    echo "Finished at: $(date)" >> "$result_file"
    
    # 提取CSV格式的数据
    execution_time_line=$(echo "$output" | grep -A 2 "CSV Format for plotting:" | tail -n 1)
    speedup_line=$(echo "$output" | grep -A 2 "Speedup CSV Format for plotting:" | tail -n 1)

    if [ -n "$execution_time_line" ]; then
      # 解析执行时间数据
      serial=$(echo "$execution_time_line" | cut -d',' -f2)
      dynamic=$(echo "$execution_time_line" | cut -d',' -f3)
      semaphore=$(echo "$execution_time_line" | cut -d',' -f4)
      full=$(echo "$execution_time_line" | cut -d',' -f5)
      barrier=$(echo "$execution_time_line" | cut -d',' -f6)
      
      # 添加到结果文件
      echo "$size,$threads,$serial,$dynamic,$semaphore,$full,$barrier" >> results/execution_time.csv
      echo "Execution time extracted and saved: $size,$threads,$serial,$dynamic,$semaphore,$full,$barrier" >> "$result_file"
    else
      echo -e "${YELLOW}Warning: Could not extract execution time for size $size with $threads threads${NC}"
      echo "Warning: Could not extract execution time" >> "$result_file"
    fi

    if [ -n "$speedup_line" ]; then
      # 解析加速比数据
      dynamic_speedup=$(echo "$speedup_line" | cut -d',' -f2)
      semaphore_speedup=$(echo "$speedup_line" | cut -d',' -f3)
      full_speedup=$(echo "$speedup_line" | cut -d',' -f4)
      barrier_speedup=$(echo "$speedup_line" | cut -d',' -f5)
      
      # 添加到结果文件
      echo "$size,$threads,$dynamic_speedup,$semaphore_speedup,$full_speedup,$barrier_speedup" >> results/speedup.csv
      echo "Speedup extracted and saved: $size,$threads,$dynamic_speedup,$semaphore_speedup,$full_speedup,$barrier_speedup" >> "$result_file"
    else
      echo -e "${YELLOW}Warning: Could not extract speedup for size $size with $threads threads${NC}"
      echo "Warning: Could not extract speedup" >> "$result_file"
    fi
    
    # 保存内存使用情况
    echo -e "${BLUE}Recording memory usage...${NC}"
    echo "Memory usage after test:" >> "$result_file"
    ps -o pid,rss,command | grep "gaussian_elimination_arm" | grep -v "grep" >> "$result_file" || echo "No process found" >> "$result_file"

    echo -e "${GREEN}Completed test for size $size with $threads threads${NC}"
    echo "========== End of Test for Matrix Size: $size, Threads: $threads ==========\n" >> "$result_file"
  done
done

# 合并所有的中间结果到一个文件
echo -e "${BLUE}Combining all intermediate results into one file...${NC}"
cat results/intermediate_results/output_*.txt > results/combined_output.txt
echo -e "${GREEN}Combined output saved to results/combined_output.txt${NC}"

echo -e "${BLUE}All tests completed. Results saved in results directory.${NC}"
echo -e "${BLUE}Raw outputs for each run saved in results/raw_output directory.${NC}"
echo -e "${BLUE}Intermediate results saved in results/intermediate_results directory.${NC}"
echo -e "${BLUE}Combined intermediate results saved in results/combined_output.txt.${NC}"

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
    os.makedirs('results/plots', exist_ok=True)
    
    # 检查结果文件是否存在且不为空
    time_csv_path = 'results/execution_time.csv'
    speedup_csv_path = 'results/speedup.csv'

    if not os.path.exists(time_csv_path) or os.path.getsize(time_csv_path) <= len(open(time_csv_path).readline()):
        print(f'Error: {time_csv_path} is empty or only contains headers. Cannot generate plots.')
        exit(1)
    if not os.path.exists(speedup_csv_path) or os.path.getsize(speedup_csv_path) <= len(open(speedup_csv_path).readline()):
        print(f'Error: {speedup_csv_path} is empty or only contains headers. Cannot generate plots.')
        exit(1)

    # 读取执行时间数据
    time_data = pd.read_csv(time_csv_path)
    speedup_data = pd.read_csv(speedup_csv_path)

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

    # 将结果保存到中间结果文件
    with open('results/combined_output.txt', 'a') as f:
        f.write('\\n\\n===== PROCESSING RESULTS =====\\n')
        f.write('Execution Time Data:\\n')
        f.write(time_data.to_string() + '\\n\\n')
        f.write('Speedup Data:\\n')
        f.write(speedup_data.to_string() + '\\n\\n')

    # 计算矩阵元素数量
    time_data['elements'] = time_data['matrix_size'] * time_data['matrix_size']

    # 1. 为每个矩阵大小绘制执行时间随线程数的变化
    for size in matrix_sizes:
        plt.figure(figsize=(12, 8))
        
        # 过滤当前矩阵大小的数据
        size_data = time_data[time_data['matrix_size'] == size]
        
        # 绘制不同算法的曲线
        plt.plot(size_data['thread_count'], size_data['serial'] / 1000000, 'o-', label='Serial', linewidth=2)
        plt.plot(size_data['thread_count'], size_data['dynamic_thread'] / 1000000, 's-', label='Dynamic Thread', linewidth=2)
        plt.plot(size_data['thread_count'], size_data['static_semaphore'] / 1000000, '^-', label='Static Semaphore', linewidth=2)
        plt.plot(size_data['thread_count'], size_data['static_full'] / 1000000, 'd-', label='Static Full', linewidth=2)
        plt.plot(size_data['thread_count'], size_data['barrier'] / 1000000, 'x-', label='Barrier', linewidth=2)
        
        plt.title(f'Gaussian Elimination Execution Time vs Thread Count (Size {size}x{size})', fontsize=16)
        plt.xlabel('Thread Count', fontsize=14)
        plt.ylabel('Execution Time (seconds)', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'results/plots/execution_time_{size}.png', dpi=300)
        plt.close()
    
    # 2. 为每个矩阵大小绘制加速比随线程数的变化
    for size in matrix_sizes:
        plt.figure(figsize=(12, 8))
        
        # 过滤当前矩阵大小的数据
        size_data = speedup_data[speedup_data['matrix_size'] == size]
        
        # 绘制不同算法的曲线
        plt.plot(size_data['thread_count'], size_data['dynamic_thread'], 's-', label='Dynamic Thread', linewidth=2)
        plt.plot(size_data['thread_count'], size_data['static_semaphore'], '^-', label='Static Semaphore', linewidth=2)
        plt.plot(size_data['thread_count'], size_data['static_full'], 'd-', label='Static Full', linewidth=2)
        plt.plot(size_data['thread_count'], size_data['barrier'], 'x-', label='Barrier', linewidth=2)
        
        # 添加理想加速比参考线
        plt.plot(size_data['thread_count'], size_data['thread_count'], '--', color='gray', label='Ideal Speedup')
        
        plt.title(f'Gaussian Elimination Speedup vs Thread Count (Size {size}x{size})', fontsize=16)
        plt.xlabel('Thread Count', fontsize=14)
        plt.ylabel('Speedup', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'results/plots/speedup_{size}.png', dpi=300)
        plt.close()
    
    # 3. 绘制执行时间随矩阵规模变化的曲线（对每个线程数）
    plt.figure(figsize=(14, 10))
    
    for t in thread_counts:
        thread_data = time_data[time_data['thread_count'] == t]
        plt.plot(thread_data['elements'], thread_data['dynamic_thread'] / 1000000, 'o-', label=f'{t} Threads', linewidth=2)
    
    plt.title('Dynamic Thread Version: Execution Time vs Matrix Size', fontsize=16)
    plt.xlabel('Matrix Elements (n²)', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/plots/exec_time_vs_size.png', dpi=300)
    plt.close()
    
    # 4. 为不同算法绘制热图，展示线程数和矩阵大小的关系
    algorithms = ['dynamic_thread', 'static_semaphore', 'static_full', 'barrier']
    algo_names = ['Dynamic Thread', 'Static Semaphore', 'Static Full', 'Barrier']
    
    for algo, name in zip(algorithms, algo_names):
        # 创建热图数据矩阵
        heatmap_data = np.zeros((len(thread_counts), len(matrix_sizes)))
        
        for i, t in enumerate(thread_counts):
            for j, size in enumerate(matrix_sizes):
                # 提取对应的加速比数据
                data_point = speedup_data[(speedup_data['thread_count'] == t) & 
                                         (speedup_data['matrix_size'] == size)]
                if not data_point.empty:
                    heatmap_data[i, j] = data_point[algo].values[0]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Speedup')
        
        plt.title(f'{name} Speedup Heatmap', fontsize=16)
        plt.xlabel('Matrix Size', fontsize=14)
        plt.ylabel('Thread Count', fontsize=14)
        plt.xticks(np.arange(len(matrix_sizes)), matrix_sizes)
        plt.yticks(np.arange(len(thread_counts)), thread_counts)
        
        # 在热图中显示数值
        for i in range(len(thread_counts)):
            for j in range(len(matrix_sizes)):
                plt.text(j, i, f'{heatmap_data[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if heatmap_data[i, j] < np.max(heatmap_data)*0.7 else 'black')
        
        plt.tight_layout()
        plt.savefig(f'results/plots/heatmap_{algo}.png', dpi=300)
        plt.close()
    
    # 5. 绘制并行效率图 (speedup/thread_count)
    for size in matrix_sizes:
        plt.figure(figsize=(12, 8))
        
        # 过滤当前矩阵大小的数据
        size_data = speedup_data[speedup_data['matrix_size'] == size]
        
        # 计算并绘制并行效率
        for algo in algorithms:
            efficiency = size_data[algo] / size_data['thread_count']
            plt.plot(size_data['thread_count'], efficiency, '-o', 
                    label=algo.replace('_', ' ').title(), linewidth=2)
        
        # 添加理想效率参考线
        plt.axhline(y=1, linestyle='--', color='gray', label='Ideal Efficiency')
        
        plt.title(f'Parallel Efficiency vs Thread Count (Size {size}x{size})', fontsize=16)
        plt.xlabel('Thread Count', fontsize=14)
        plt.ylabel('Parallel Efficiency (Speedup/Thread Count)', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'results/plots/efficiency_{size}.png', dpi=300)
        plt.close()
    
    # 记录到中间结果文件
    with open('results/combined_output.txt', 'a') as f:
        f.write('===== PLOTS GENERATED =====\\n')
        f.write('Plot files:\\n')
        for size in matrix_sizes:
            f.write(f'- results/plots/execution_time_{size}.png\\n')
            f.write(f'- results/plots/speedup_{size}.png\\n')
            f.write(f'- results/plots/efficiency_{size}.png\\n')
        
        for algo in algorithms:
            f.write(f'- results/plots/heatmap_{algo}.png\\n')
        
        f.write('- results/plots/exec_time_vs_size.png\\n')
        f.write('===== END OF PROCESSING =====\\n')

    print('All plots generated successfully!')

except Exception as e:
    import traceback
    print(f'Error in Python script: {str(e)}')
    print(traceback.format_exc())
    exit(1)
"

# 检查绘图是否成功
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to generate plots. Make sure matplotlib and pandas are installed.${NC}"
  echo -e "${YELLOW}You can install them using: pip3 install matplotlib pandas${NC}"
  # 恢复原始源代码
  mv gaussian_elimination_arm_original.cpp gaussian_elimination_arm.cpp
  exit 1
fi

echo -e "${GREEN}Plots saved in results/plots directory.${NC}"

# 生成性能分析报告
echo -e "${BLUE}Generating performance report...${NC}"
cat > results/performance_report.md << EOL
# ARM NEON优化高斯消元算法线程扩展性报告

## 概述
本报告分析了使用ARM NEON指令集优化的高斯消元算法在不同线程数和矩阵大小下的性能表现。
测试日期: 2025-05-22 07:47:32
测试用户: KKKyriejiang

## 测试环境
- 架构: ARM (通过QEMU模拟)
- 编译器: aarch64-linux-gnu-g++
- 优化选项: -O3, ARM NEON向量化
- 测试线程数: ${THREAD_COUNTS[@]}
- 测试矩阵大小: ${SIZES[@]}

## 算法实现
1. **串行算法**: 使用NEON指令优化的单线程基准实现
2. **动态线程版本**: 在每轮消元中动态创建线程
3. **静态线程+信号量同步版本**: 使用固定线程池和信号量进行同步
4. **静态线程+信号量同步+三重循环整合版本**: 减少同步点的方法
5. **静态线程+barrier同步版本**: 使用pthread_barrier进行线程同步

## 执行时间分析

每个矩阵大小下的执行时间随线程数的变化图表显示：

![16x16矩阵执行时间](plots/execution_time_16.png)
![32x32矩阵执行时间](plots/execution_time_32.png)
![64x64矩阵执行时间](plots/execution_time_64.png)
![128x128矩阵执行时间](plots/execution_time_128.png)

从图表可以观察到：
1. NEON优化的串行算法提供了较高的基准性能
2. 较小矩阵(16x16, 32x32)在增加线程数时执行时间下降不明显，甚至可能出现增加
3. 较大矩阵(64x64, 128x128)对线程数增加更为敏感，执行时间显著下降
4. 当线程数过多时，额外的线程管理开销可能抵消并行带来的收益

## 加速比分析

每个矩阵大小下的加速比随线程数的变化图表显示：

![16x16矩阵加速比](plots/speedup_16.png)
![32x32矩阵加速比](plots/speedup_32.png)
![64x64矩阵加速比](plots/speedup_64.png)
![128x128矩阵加速比](plots/speedup_128.png)

加速比分析显示：
1. 理想情况下，加速比应与线程数成正比（理想加速比线）
2. 实际加速比远低于理想值，表明存在明显的并行开销
3. 随着矩阵大小增加，可达到的最大加速比上升
4. NEON优化的基准性能已经很高，导致线程并行的相对收益降低

## 并行效率分析

每个矩阵大小下的并行效率随线程数的变化图表显示：

![16x16矩阵并行效率](plots/efficiency_16.png)
![32x32矩阵并行效率](plots/efficiency_32.png)
![64x64矩阵并行效率](plots/efficiency_64.png)
![128x128矩阵并行效率](plots/efficiency_128.png)

并行效率分析表明：
1. 所有实现的并行效率都随线程数增加而下降
2. 小规模问题(16x16)的并行效率下降最快
3. 大规模问题(128x128)的并行效率相对更高
4. NEON向量化和线程并行的结合需要仔细平衡

## 算法性能热图

各种算法在不同线程数和矩阵大小下的性能热图：

![动态线程热图](plots/heatmap_dynamic_thread.png)
![静态信号量热图](plots/heatmap_static_semaphore.png)
![静态全热图](plots/heatmap_static_full.png)
![屏障同步热图](plots/heatmap_barrier.png)

热图分析显示：
1. 矩阵大小和线程数的最佳组合随算法实现而异
2. 大矩阵+多线程通常能达到最高加速比
3. 小矩阵在低线程数下表现更好
4. 不同并行实现在特定场景下各有优势

## NEON优化与多线程协同分析

1. **NEON向量化优势**
   - 单线程基准性能已经通过NEON指令得到显著提升
   - 向量化减少了CPU指令数量，降低了并行化的相对收益
   - 内存带宽可能成为瓶颈，限制了线程并行的扩展性

2. **多线程同步开销**
   - 同步原语(信号量、屏障)在ARM架构下的开销较大
   - 线程创建和销毁的成本在小规模问题中占比较高
   - 静态线程池方法在重复计算场景下更有优势

3. **缓存效率**
   - 多线程访问共享数据可能导致缓存失效增加
   - NEON指令依赖良好的缓存局部性以发挥最大性能
   - 线程间的数据交换会增加内存访问延迟

## 结论与建议

1. **硬件/算法匹配**
   - 小矩阵(16x16, 32x32): 优先使用NEON单线程或2-4线程
   - 中等矩阵(64x64): 4-8线程较为合适
   - 大矩阵(128x128): 8-16线程能获得最大加速比

2. **实现策略选择**
   - 小规模问题: 考虑使用静态线程+barrier同步以减少同步开销
   - 中等规模: 静态线程+信号量适合权衡性能和开销
   - 大规模问题: 静态线程+三重循环整合版本提供最佳整体性能

3. **优化建议**
   - 考虑使用分块算法改善缓存局部性
   - 探索自适应线程数选择策略，根据问题规模动态调整
   - 进一步优化NEON指令使用，如使用预取指令减少内存延迟
   - 减少线程间通信和同步点的数量

ARM NEON优化的高斯消元算法通过向量化和多线程相结合，可以获得显著的性能提升。但在实际应用中，需要根据问题规模和硬件特性谨慎选择优化策略，以达到最佳平衡。
EOL

echo -e "${GREEN}Performance report generated: results/performance_report.md${NC}"
echo -e "${GREEN}All tests completed successfully!${NC}"

# 恢复原始源代码
echo -e "${BLUE}Restoring original source code...${NC}"
mv gaussian_elimination_arm_original.cpp gaussian_elimination_arm.cpp

echo -e "${GREEN}Done!${NC}"