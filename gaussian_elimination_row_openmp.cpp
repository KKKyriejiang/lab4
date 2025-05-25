#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <arm_neon.h>
#include <omp.h>
using namespace std;

// 矩阵大小
int n = 0;
// 默认线程数量
int NUM_THREADS = 4;
// 矩阵数据
float **matrix = nullptr;

// 用于计时
struct timeval start_time, end_time;
long long execution_time;

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

// 1. 基本OpenMP版本 - 按行划分
void gaussEliminationDynamicThread() {
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        division_neon(k);
        
        // 使用OpenMP并行化消去操作
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = k + 1; i < n; i++) {
            elimination_neon(k, i);
        }
    }
}

// 2. 静态线程+信号量同步版本的OpenMP实现 - 使用单一并行区域
void gaussEliminationStaticSemaphore() {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k++) {
            // 只让一个线程执行归一化
            #pragma omp single
            {
                division_neon(k);
            }
            
            // 并行化消去操作，隐式同步
            #pragma omp for
            for (int i = k + 1; i < n; i++) {
                elimination_neon(k, i);
            }
        }
    }
}

// 3. 静态线程 + 更多优化 - 使用nowait减少同步开销
void gaussEliminationStaticFull() {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k++) {
            // 只让一个线程执行归一化
            #pragma omp single
            {
                division_neon(k);
            }
            // single隐含barrier，确保division完成
            
            // 使用nowait避免不必要的同步
            #pragma omp for nowait
            for (int i = k + 1; i < n; i++) {
                elimination_neon(k, i);
            }
            
            // 使用显式同步确保当前迭代完成
            #pragma omp barrier
        }
    }
}

// 4. 静态线程+动态调度版本 - 使用调度策略优化负载均衡
void gaussEliminationBarrier() {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k++) {
            // 只让一个线程执行归一化
            #pragma omp single
            {
                division_neon(k);
            }
            
            // 使用dynamic调度提高负载均衡
            #pragma omp for schedule(dynamic, 16)
            for (int i = k + 1; i < n; i++) {
                elimination_neon(k, i);
            }
        }
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

// 计算执行时间（微秒）
long long get_execution_time() {
    return (end_time.tv_sec - start_time.tv_sec) * 1000000LL + (end_time.tv_usec - start_time.tv_usec);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <matrix_size> [num_threads]" << endl;
        return -1;
    }
    
    // 设置矩阵大小
    n = atoi(argv[1]);
    if (n <= 0) {
        cout << "Invalid matrix size" << endl;
        return -1;
    }
    
    // 设置线程数（如果提供）
    if (argc >= 3) {
        int threads = atoi(argv[2]);
        if (threads > 0) {
            NUM_THREADS = threads;
        } else {
            cout << "Invalid number of threads, using default: " << NUM_THREADS << endl;
        }
    }
    
    // 设置OpenMP线程数
    omp_set_num_threads(NUM_THREADS);
    
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Number of threads: " << NUM_THREADS << endl;
    
    // 初始化随机数生成器
    srand(42);
    
    // 初始化矩阵
    init_matrix();
    
    // 保存原始矩阵
    float** original_matrix = save_result();
    
    // 运行并测试串行版本
    gettimeofday(&start_time, NULL);
    gaussEliminationSerial();
    gettimeofday(&end_time, NULL);
    long long serial_time = get_execution_time();
    float** serial_result = save_result();
    
    cout << "Serial version execution time: " << serial_time << " us" << endl;
    
    // 运行并测试版本1：基本OpenMP版本 - 按行划分
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationDynamicThread();
    gettimeofday(&end_time, NULL);
    long long dynamic_thread_time = get_execution_time();
    
    cout << "Dynamic Thread version (OpenMP) execution time: " << dynamic_thread_time << " us" << endl;
    cout << "Dynamic Thread version (OpenMP) speedup: " << (float)serial_time / dynamic_thread_time << endl;
    cout << "Dynamic Thread version (OpenMP) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本2：静态线程+单一并行区域版本 - 按行划分
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticSemaphore();
    gettimeofday(&end_time, NULL);
    long long static_semaphore_time = get_execution_time();
    
    cout << "Static Semaphore version (OpenMP) execution time: " << static_semaphore_time << " us" << endl;
    cout << "Static Semaphore version (OpenMP) speedup: " << (float)serial_time / static_semaphore_time << endl;
    cout << "Static Semaphore version (OpenMP) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本3：静态线程+nowait优化版本 - 按行划分
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticFull();
    gettimeofday(&end_time, NULL);
    long long static_full_time = get_execution_time();
    
    cout << "Static Full Thread version (OpenMP) execution time: " << static_full_time << " us" << endl;
    cout << "Static Full Thread version (OpenMP) speedup: " << (float)serial_time / static_full_time << endl;
    cout << "Static Full Thread version (OpenMP) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本4：静态线程+动态调度版本 - 按行划分
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationBarrier();
    gettimeofday(&end_time, NULL);
    long long barrier_time = get_execution_time();
    
    cout << "Dynamic Schedule version (OpenMP) execution time: " << barrier_time << " us" << endl;
    cout << "Dynamic Schedule version (OpenMP) speedup: " << (float)serial_time / barrier_time << endl;
    cout << "Dynamic Schedule version (OpenMP) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 输出CSV格式的执行时间
    cout << "\nCSV Format for plotting:\n";
    cout << "matrix_size,serial,dynamic_thread,static_semaphore,static_full,barrier\n";
    cout << n << "," << serial_time << "," << dynamic_thread_time << "," 
         << static_semaphore_time << "," << static_full_time << "," << barrier_time << endl;
    
    // 输出CSV格式的加速比
    cout << "\nSpeedup CSV Format for plotting:\n";
    cout << "matrix_size,dynamic_thread,static_semaphore,static_full,barrier\n";
    cout << n << "," << (float)serial_time / dynamic_thread_time << "," 
         << (float)serial_time / static_semaphore_time << "," 
         << (float)serial_time / static_full_time << "," 
         << (float)serial_time / barrier_time << endl;
    
    // 释放内存
    free_result(original_matrix);
    free_result(serial_result);
    free_matrix();
    
    return 0;
}