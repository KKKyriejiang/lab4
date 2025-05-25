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
// 线程数量
const int NUM_THREADS = 4;
// 缓存优化的分块大小
const int BLOCK_SIZE = 64; // 可根据实际缓存大小调整
// 矩阵数据 - 使用一维数组存储，提高缓存命中率
float *matrix = nullptr;

// 用于计时
struct timeval start_time, end_time;
long long execution_time;

// 同步所需的信号量和屏障
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];
sem_t sem_leader;
sem_t sem_division[NUM_THREADS-1];
sem_t sem_elimination[NUM_THREADS-1];
pthread_barrier_t barrier_division;
pthread_barrier_t barrier_elimination;

// 线程参数结构体
typedef struct {
    int k;       // 当前消去的行
    int t_id;    // 线程ID
    int* row_indices; // 存储每个线程负责的行索引
    int num_rows;     // 线程负责的行数
} threadParam_t;

// 访问矩阵中(i,j)元素的宏，提高代码可读性
#define MATRIX(i, j) matrix[(i) * n + (j)]

// 初始化矩阵，使用随机数填充 - 使用一维数组连续存储，提高缓存命中率
void init_matrix() {
    matrix = new float[n * n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            MATRIX(i, j) = rand() % 100 + 1;
        }
    }
    // 保证矩阵可逆
    for (int i = 0; i < n; i++) {
        MATRIX(i, i) += n * 10;
    }
}

// 释放矩阵内存
void free_matrix() {
    delete[] matrix;
}

// NEON向量化的行归一化函数 - 缓存优化版本
void division_neon(int k) {
    float pivot = MATRIX(k, k);
    float32x4_t vt = vdupq_n_f32(pivot);
    int j = k + 1;
    
    // 预取后续数据以减少缓存缺失
    __builtin_prefetch(&MATRIX(k, j + 16), 0, 3);
    
    // 使用NEON指令进行向量化除法，每次处理4个元素
    for (; j + 3 < n; j += 4) {
        // 预取后续数据，减少缓存缺失
        __builtin_prefetch(&MATRIX(k, j + 16), 0, 3);
        
        float32x4_t va = vld1q_f32(&MATRIX(k, j));
        va = vdivq_f32(va, vt);
        vst1q_f32(&MATRIX(k, j), va);
    }
    
    // 处理剩余的元素
    for (; j < n; j++) {
        MATRIX(k, j) = MATRIX(k, j) / pivot;
    }
    
    MATRIX(k, k) = 1.0;
}

// NEON向量化的消去函数 - 缓存优化版本
void elimination_neon(int k, int i) {
    float multiplier = MATRIX(i, k);
    float32x4_t vaik = vdupq_n_f32(multiplier);
    int j = k + 1;
    
    // 预取k行和i行的数据
    __builtin_prefetch(&MATRIX(k, j + 16), 0, 3);
    __builtin_prefetch(&MATRIX(i, j + 16), 1, 3);
    
    // 使用NEON指令进行向量化消去，每次处理4个元素
    for (; j + 3 < n; j += 4) {
        // 预取后续数据
        __builtin_prefetch(&MATRIX(k, j + 16), 0, 3);
        __builtin_prefetch(&MATRIX(i, j + 16), 1, 3);
        
        float32x4_t vakj = vld1q_f32(&MATRIX(k, j));
        float32x4_t vaij = vld1q_f32(&MATRIX(i, j));
        float32x4_t vx = vmulq_f32(vaik, vakj);
        vaij = vsubq_f32(vaij, vx);
        vst1q_f32(&MATRIX(i, j), vaij);
    }
    
    // 处理剩余的元素
    for (; j < n; j++) {
        MATRIX(i, j) = MATRIX(i, j) - multiplier * MATRIX(k, j);
    }
    
    MATRIX(i, k) = 0;
}

// 分块优化的串行高斯消去算法
void gaussEliminationSerialBlocked() {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int kk = 0; kk < num_blocks; kk++) {
        int k_start = kk * BLOCK_SIZE;
        int k_end = min(n, k_start + BLOCK_SIZE);
        
        // 对当前块内行进行处理
        for (int k = k_start; k < k_end; k++) {
            // 归一化当前行
            division_neon(k);
            
            // 更新当前块内的其他行
            for (int i = k + 1; i < k_end; i++) {
                elimination_neon(k, i);
            }
            
            // 更新当前块右侧的所有块
            for (int jj = kk + 1; jj < num_blocks; jj++) {
                int j_start = jj * BLOCK_SIZE;
                int j_end = min(n, j_start + BLOCK_SIZE);
                
                // 对其他块进行更新
                for (int i = k + 1; i < k_end; i++) {
                    float multiplier = MATRIX(i, k);
                    
                    for (int j = j_start; j < j_end; j++) {
                        MATRIX(i, j) -= multiplier * MATRIX(k, j);
                    }
                }
            }
            
            // 更新其他块
            for (int ii = kk + 1; ii < num_blocks; ii++) {
                int i_start = ii * BLOCK_SIZE;
                int i_end = min(n, i_start + BLOCK_SIZE);
                
                // 使用当前行更新其他块的行
                for (int i = i_start; i < i_end; i++) {
                    elimination_neon(k, i);
                }
                
                // 更新对角线上和对角线右侧的块
                for (int jj = kk + 1; jj < num_blocks; jj++) {
                    int j_start = jj * BLOCK_SIZE;
                    int j_end = min(n, j_start + BLOCK_SIZE);
                    
                    // 更新块中的元素
                    for (int i = i_start; i < i_end; i++) {
                        float multiplier = MATRIX(i, k);
                        
                        for (int j = j_start; j < j_end; j++) {
                            MATRIX(i, j) -= multiplier * MATRIX(k, j);
                        }
                    }
                }
            }
        }
    }
}

// 标准（非分块）串行高斯消去算法
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

// 计算每个线程应该处理的行数和行索引 - 缓存优化版本
void calculateRowDistribution(int k, int* row_per_thread, int** row_indices) {
    int remaining_rows = n - k - 1;
    
    // 初始化每个线程处理的行数
    for (int t = 0; t < NUM_THREADS; t++) {
        row_per_thread[t] = 0;
    }
    
    // 缓存优化：分配连续块给每个线程，而不是交错分配
    int chunk_size = remaining_rows / NUM_THREADS;
    int remainder = remaining_rows % NUM_THREADS;
    
    for (int t = 0; t < NUM_THREADS; t++) {
        row_per_thread[t] = chunk_size + (t < remainder ? 1 : 0);
    }
    
    // 分配行索引数组空间
    for (int t = 0; t < NUM_THREADS; t++) {
        row_indices[t] = new int[row_per_thread[t]];
    }
    
    // 填充行索引 - 按连续块分配
    int row_start = k + 1;
    for (int t = 0; t < NUM_THREADS; t++) {
        for (int i = 0; i < row_per_thread[t]; i++) {
            row_indices[t][i] = row_start++;
        }
    }
}

// 1. 动态线程版本的消去线程函数 - 按行划分 - 缓存优化版本
void* dynamicThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    
    // 预取k行关键数据
    __builtin_prefetch(&MATRIX(k, k), 0, 3);
    __builtin_prefetch(&MATRIX(k, k + 4), 0, 3);
    __builtin_prefetch(&MATRIX(k, k + 8), 0, 3);
    
    // 处理线程分配到的所有行
    for (int idx = 0; idx < p->num_rows; idx++) {
        int i = p->row_indices[idx];
        
        // 预取当前行的数据
        __builtin_prefetch(&MATRIX(i, k), 0, 3);
        __builtin_prefetch(&MATRIX(i, k + 4), 1, 3);
        
        elimination_neon(k, i);
    }
    
    pthread_exit(NULL);
}

// 1. 动态线程版本的高斯消去算法 - 按行划分 - 缓存优化版本
void gaussEliminationDynamicThread() {
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        division_neon(k);
        
        // 如果是最后一行，没有需要消去的行了
        if (k == n - 1) continue;
        
        // 计算每个线程需要处理的行
        int* row_per_thread = new int[NUM_THREADS];
        int** row_indices = new int*[NUM_THREADS];
        calculateRowDistribution(k, row_per_thread, row_indices);
        
        // 创建线程
        pthread_t* handles = new pthread_t[NUM_THREADS];
        threadParam_t* param = new threadParam_t[NUM_THREADS];
        
        // 创建并启动线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            // 如果线程没有分配到行，则跳过
            if (row_per_thread[t_id] == 0) continue;
            
            param[t_id].k = k;
            param[t_id].t_id = t_id;
            param[t_id].row_indices = row_indices[t_id];
            param[t_id].num_rows = row_per_thread[t_id];
            
            pthread_create(&handles[t_id], NULL, dynamicThreadFunc, &param[t_id]);
        }
        
        // 等待所有线程完成
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            if (row_per_thread[t_id] > 0) {
                pthread_join(handles[t_id], NULL);
            }
        }
        
        // 释放资源
        for (int t = 0; t < NUM_THREADS; t++) {
            if (row_per_thread[t] > 0) {
                delete[] row_indices[t];
            }
        }
        delete[] row_indices;
        delete[] row_per_thread;
        delete[] handles;
        delete[] param;
    }
}

// 2. 静态线程+信号量同步版本的线程函数 - 按行划分 - 缓存优化版本
void* staticSemaphoreThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]); // 等待主线程通知开始工作
        
        // 缓存优化：使用块划分而不是循环划分，保持内存局部性
        int chunk_size = (n - k - 1) / NUM_THREADS;
        int remainder = (n - k - 1) % NUM_THREADS;
        int start_row = k + 1 + t_id * chunk_size + min(t_id, remainder);
        int end_row = start_row + chunk_size + (t_id < remainder ? 1 : 0);
        
        // 预取k行数据
        __builtin_prefetch(&MATRIX(k, k), 0, 3);
        __builtin_prefetch(&MATRIX(k, k + 4), 0, 3);
        
        // 处理分配的连续行块
        for (int i = start_row; i < end_row; i++) {
            // 预取当前行的数据
            __builtin_prefetch(&MATRIX(i, k), 0, 3);
            __builtin_prefetch(&MATRIX(i, k + 4), 1, 3);
            
            elimination_neon(k, i);
        }
        
        sem_post(&sem_main); // 通知主线程已完成
        sem_wait(&sem_workerend[t_id]); // 等待所有工作线程完成当前轮次
    }
    
    pthread_exit(NULL);
}

// 2. 静态线程+信号量同步版本的高斯消去算法 - 按行划分 - 缓存优化版本
void gaussEliminationStaticSemaphore() {
    // 初始化信号量
    sem_init(&sem_main, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    
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
    
    // 销毁信号量
    sem_destroy(&sem_main);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
}

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的线程函数 - 按行划分 - 缓存优化版本
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
        
        // 缓存优化：使用块划分而不是循环划分，保持内存局部性
        int chunk_size = (n - k - 1) / NUM_THREADS;
        int remainder = (n - k - 1) % NUM_THREADS;
        int start_row = k + 1 + t_id * chunk_size + min(t_id, remainder);
        int end_row = start_row + chunk_size + (t_id < remainder ? 1 : 0);
        
        // 预取k行数据
        __builtin_prefetch(&MATRIX(k, k), 0, 3);
        __builtin_prefetch(&MATRIX(k, k + 4), 0, 3);
        
        // 处理分配的连续行块
        for (int i = start_row; i < end_row; i++) {
            // 预取当前行的数据
            __builtin_prefetch(&MATRIX(i, k), 0, 3);
            __builtin_prefetch(&MATRIX(i, k + 4), 1, 3);
            
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

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的高斯消去算法 - 按行划分 - 缓存优化版本
void gaussEliminationStaticFull() {
    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        sem_init(&sem_division[i], 0, 0);
        sem_init(&sem_elimination[i], 0, 0);
    }
    
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
    
    // 销毁信号量
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        sem_destroy(&sem_division[i]);
        sem_destroy(&sem_elimination[i]);
    }
}

// 4. 静态线程+barrier同步版本的线程函数 - 按行划分 - 缓存优化版本
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
        
        // 缓存优化：使用块划分而不是循环划分，保持内存局部性
        int chunk_size = (n - k - 1) / NUM_THREADS;
        int remainder = (n - k - 1) % NUM_THREADS;
        int start_row = k + 1 + t_id * chunk_size + min(t_id, remainder);
        int end_row = start_row + chunk_size + (t_id < remainder ? 1 : 0);
        
        // 预取k行数据
        __builtin_prefetch(&MATRIX(k, k), 0, 3);
        __builtin_prefetch(&MATRIX(k, k + 4), 0, 3);
        
        // 处理分配的连续行块
        for (int i = start_row; i < end_row; i++) {
            // 预取当前行的数据
            __builtin_prefetch(&MATRIX(i, k), 0, 3);
            __builtin_prefetch(&MATRIX(i, k + 4), 1, 3);
            
            elimination_neon(k, i);
        }
        
        // 使用屏障同步，确保所有消去操作完成
        pthread_barrier_wait(&barrier_elimination);
    }
    
    pthread_exit(NULL);
}

// 4. 静态线程+barrier同步版本的高斯消去算法 - 按行划分 - 缓存优化版本
void gaussEliminationBarrier() {
    // 初始化屏障
    pthread_barrier_init(&barrier_division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_elimination, NULL, NUM_THREADS);
    
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
    
    // 销毁屏障
    pthread_barrier_destroy(&barrier_division);
    pthread_barrier_destroy(&barrier_elimination);
}

// 检查结果是否正确（与串行算法比较）- 缓存优化版本
bool check_result(float* result_matrix) {
    // 创建一个拷贝用于串行计算
    float* temp_matrix = new float[n * n];
    memcpy(temp_matrix, matrix, n * n * sizeof(float));
    
    // 运行标准串行算法
    for (int k = 0; k < n; k++) {
        // 归一化
        for (int j = k + 1; j < n; j++) {
            temp_matrix[k * n + j] /= temp_matrix[k * n + k];
        }
        temp_matrix[k * n + k] = 1.0;
        
        // 消去
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                temp_matrix[i * n + j] -= temp_matrix[i * n + k] * temp_matrix[k * n + j];
            }
            temp_matrix[i * n + k] = 0;
        }
    }
    
    // 比较结果
    bool correct = true;
    for (int i = 0; i < n && correct; i++) {
        for (int j = 0; j < n && correct; j++) {
            if (fabs(result_matrix[i * n + j] - temp_matrix[i * n + j]) > 1e-4) {
                correct = false;
                cout << "Difference at [" << i << "][" << j << "]: " 
                     << result_matrix[i * n + j] << " vs " << temp_matrix[i * n + j] << endl;
            }
        }
    }
    
    // 释放临时矩阵
    delete[] temp_matrix;
    
    return correct;
}

// 保存结果 - 缓存优化版本
float* save_result() {
    float* result = new float[n * n];
    memcpy(result, matrix, n * n * sizeof(float));
    return result;
}

// 恢复原始矩阵 - 缓存优化版本
void restore_matrix(float* result) {
    memcpy(matrix, result, n * n * sizeof(float));
}

// 释放结果矩阵 - 缓存优化版本
void free_result(float* result) {
    delete[] result;
}

// 计算执行时间
long long get_execution_time() {
    return (end_time.tv_sec - start_time.tv_sec) * 1000000LL + (end_time.tv_usec - start_time.tv_usec);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return -1;
    }
    
    // 设置矩阵大小
    n = atoi(argv[1]);
    if (n <= 0) {
        cout << "Invalid matrix size" << endl;
        return -1;
    }
    
    // 初始化随机数生成器
    srand(42);
    
    // 初始化矩阵
    init_matrix();
    
    // 保存原始矩阵
    float* original_matrix = save_result();
    
    // 运行并测试串行版本
    gettimeofday(&start_time, NULL);
    gaussEliminationSerial();
    gettimeofday(&end_time, NULL);
    long long serial_time = get_execution_time();
    float* serial_result = save_result();
    
    cout << "Serial version execution time: " << serial_time << " us" << endl;
    
    // 运行并测试分块优化的串行版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationSerialBlocked();
    gettimeofday(&end_time, NULL);
    long long serial_blocked_time = get_execution_time();
    
    cout << "Serial Blocked version execution time: " << serial_blocked_time << " us" << endl;
    cout << "Serial Blocked version speedup: " << (float)serial_time / serial_blocked_time << endl;
    cout << "Serial Blocked version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本1：动态线程版本 - 按行划分 - 缓存优化
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationDynamicThread();
    gettimeofday(&end_time, NULL);
    long long dynamic_thread_time = get_execution_time();
    
    cout << "Dynamic Thread version (Row-Based, Cache Optimized) execution time: " << dynamic_thread_time << " us" << endl;
    cout << "Dynamic Thread version (Row-Based, Cache Optimized) speedup: " << (float)serial_time / dynamic_thread_time << endl;
    cout << "Dynamic Thread version (Row-Based, Cache Optimized) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本2：静态线程+信号量同步版本 - 按行划分 - 缓存优化
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticSemaphore();
    gettimeofday(&end_time, NULL);
    long long static_semaphore_time = get_execution_time();
    
    cout << "Static Semaphore version (Row-Based, Cache Optimized) execution time: " << static_semaphore_time << " us" << endl;
    cout << "Static Semaphore version (Row-Based, Cache Optimized) speedup: " << (float)serial_time / static_semaphore_time << endl;
    cout << "Static Semaphore version (Row-Based, Cache Optimized) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本3：静态线程+信号量同步+三重循环全部纳入线程函数版本 - 按行划分 - 缓存优化
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticFull();
    gettimeofday(&end_time, NULL);
    long long static_full_time = get_execution_time();
    
    cout << "Static Full Thread version (Row-Based, Cache Optimized) execution time: " << static_full_time << " us" << endl;
    cout << "Static Full Thread version (Row-Based, Cache Optimized) speedup: " << (float)serial_time / static_full_time << endl;
    cout << "Static Full Thread version (Row-Based, Cache Optimized) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本4：静态线程+barrier同步版本 - 按行划分 - 缓存优化
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationBarrier();
    gettimeofday(&end_time, NULL);
    long long barrier_time = get_execution_time();
    
    cout << "Barrier version (Row-Based, Cache Optimized) execution time: " << barrier_time << " us" << endl;
    cout << "Barrier version (Row-Based, Cache Optimized) speedup: " << (float)serial_time / barrier_time << endl;
    cout << "Barrier version (Row-Based, Cache Optimized) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 输出CSV格式的执行时间
    cout << "\nCSV Format for plotting:\n";
    cout << "matrix_size,serial,serial_blocked,dynamic_thread,static_semaphore,static_full,barrier\n";
    cout << n << "," << serial_time << "," << serial_blocked_time << "," << dynamic_thread_time << "," 
         << static_semaphore_time << "," << static_full_time << "," << barrier_time << endl;
    
    // 输出CSV格式的加速比
    cout << "\nSpeedup CSV Format for plotting:\n";
    cout << "matrix_size,serial_blocked,dynamic_thread,static_semaphore,static_full,barrier\n";
    cout << n << "," << (float)serial_time / serial_blocked_time << "," 
         << (float)serial_time / dynamic_thread_time << "," 
         << (float)serial_time / static_semaphore_time << "," 
         << (float)serial_time / static_full_time << "," 
         << (float)serial_time / barrier_time << endl;
    
    // 释放内存
    free_result(original_matrix);
    free_result(serial_result);
    free_matrix();
    
    return 0;
}