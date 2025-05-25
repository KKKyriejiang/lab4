#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h> // AVX指令集头文件
#include <cmath>
using namespace std;

// 矩阵大小
int n = 0;
// 线程数量
const int NUM_THREADS = 4;
// 矩阵数据
float **matrix = nullptr;

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

// 初始化矩阵，使用随机数填充，确保内存对齐
void init_matrix() {
    matrix = new float*[n];
    for (int i = 0; i < n; i++) {
        // 使用_mm_malloc确保32字节对齐（AVX需要）
        matrix[i] = (float*)_mm_malloc(n * sizeof(float), 32);
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
        _mm_free(matrix[i]); // 使用_mm_free释放由_mm_malloc分配的内存
    }
    delete[] matrix;
}

// AVX向量化的行归一化函数
void division_avx(int k) {
    __m256 vt = _mm256_set1_ps(matrix[k][k]);
    int j = k + 1;
    
    // 使用AVX指令进行向量化除法（每次处理8个float）
    for (; j + 7 < n; j += 8) {
        __m256 va = _mm256_loadu_ps(&matrix[k][j]);
        va = _mm256_div_ps(va, vt);
        _mm256_storeu_ps(&matrix[k][j], va);
    }
    
    // 处理剩余的元素
    for (; j < n; j++) {
        matrix[k][j] = matrix[k][j] / matrix[k][k];
    }
    
    matrix[k][k] = 1.0;
}

// AVX向量化的消去函数
void elimination_avx(int k, int i) {
    __m256 vaik = _mm256_set1_ps(matrix[i][k]);
    int j = k + 1;
    
    // 使用AVX指令进行向量化消去（每次处理8个float）
    for (; j + 7 < n; j += 8) {
        __m256 vakj = _mm256_loadu_ps(&matrix[k][j]);
        __m256 vaij = _mm256_loadu_ps(&matrix[i][j]);
        __m256 vx = _mm256_mul_ps(vaik, vakj);
        vaij = _mm256_sub_ps(vaij, vx);
        _mm256_storeu_ps(&matrix[i][j], vaij);
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
        division_avx(k);
        
        // 消去后续各行
        for (int i = k + 1; i < n; i++) {
            elimination_avx(k, i);
        }
    }
}

// 计算每个线程应该处理的行数和行索引
void calculateRowDistribution(int k, int* row_per_thread, int** row_indices) {
    int remaining_rows = n - k - 1;
    
    // 初始化每个线程处理的行数
    for (int t = 0; t < NUM_THREADS; t++) {
        row_per_thread[t] = 0;
    }
    
    // 分配行给线程，使用连续的行分配方式
    for (int i = 0; i < remaining_rows; i++) {
        int thread_id = i % NUM_THREADS;
        row_per_thread[thread_id]++;
    }
    
    // 分配行索引数组空间
    for (int t = 0; t < NUM_THREADS; t++) {
        row_indices[t] = new int[row_per_thread[t]];
    }
    
    // 填充行索引
    int* row_counters = new int[NUM_THREADS]();  // 初始化为0
    for (int i = 0; i < remaining_rows; i++) {
        int actual_row = k + 1 + i;
        int thread_id = i % NUM_THREADS;
        row_indices[thread_id][row_counters[thread_id]++] = actual_row;
    }
    
    delete[] row_counters;
}

// 1. 动态线程版本的消去线程函数 - 按行划分
void* dynamicThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    
    // 处理线程分配到的所有行
    for (int idx = 0; idx < p->num_rows; idx++) {
        int i = p->row_indices[idx];
        elimination_avx(k, i);
    }
    
    pthread_exit(NULL);
    return NULL;
}

// 1. 动态线程版本的高斯消去算法 - 按行划分
void gaussEliminationDynamicThread() {
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        division_avx(k);
        
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

// 2. 静态线程+信号量同步版本的线程函数 - 按行划分
void* staticSemaphoreThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]); // 等待主线程通知开始工作
        
        // 按行划分，处理分配给本线程的行
        int chunk_size = (n - k - 1) / NUM_THREADS + ((n - k - 1) % NUM_THREADS > t_id ? 1 : 0);
        int start_row = k + 1 + t_id * ((n - k - 1) / NUM_THREADS) + min(t_id, (n - k - 1) % NUM_THREADS);
        
        for (int i = 0; i < chunk_size; i++) {
            elimination_avx(k, start_row + i);
        }
        
        sem_post(&sem_main); // 通知主线程已完成
        sem_wait(&sem_workerend[t_id]); // 等待所有工作线程完成当前轮次
    }
    
    pthread_exit(NULL);
    return NULL;
}

// 2. 静态线程+信号量同步版本的高斯消去算法 - 按行划分
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
        division_avx(k);
        
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

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的线程函数 - 按行划分
void* staticFullThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // 线程0负责归一化操作
        if (t_id == 0) {
            division_avx(k);
        } else {
            sem_wait(&sem_division[t_id-1]); // 非0线程等待归一化完成
        }
        
        // 线程0通知其他线程归一化已完成
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_division[i]);
            }
        }
        
        // 按行划分，每个线程处理连续的一块行
        int chunk_size = (n - k - 1) / NUM_THREADS + ((n - k - 1) % NUM_THREADS > t_id ? 1 : 0);
        int start_row = k + 1 + t_id * ((n - k - 1) / NUM_THREADS) + min(t_id, (n - k - 1) % NUM_THREADS);
        
        for (int i = 0; i < chunk_size; i++) {
            elimination_avx(k, start_row + i);
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
    return NULL;
}

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的高斯消去算法 - 按行划分
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

// 4. 静态线程+barrier同步版本的线程函数 - 按行划分
void* staticBarrierThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // 线程0负责归一化操作
        if (t_id == 0) {
            division_avx(k);
        }
        
        // 使用屏障同步，确保归一化完成
        pthread_barrier_wait(&barrier_division);
        
        // 按行划分，每个线程处理连续的一块行
        int chunk_size = (n - k - 1) / NUM_THREADS + ((n - k - 1) % NUM_THREADS > t_id ? 1 : 0);
        int start_row = k + 1 + t_id * ((n - k - 1) / NUM_THREADS) + min(t_id, (n - k - 1) % NUM_THREADS);
        
        for (int i = 0; i < chunk_size; i++) {
            elimination_avx(k, start_row + i);
        }
        
        // 使用屏障同步，确保所有消去操作完成
        pthread_barrier_wait(&barrier_elimination);
    }
    
    pthread_exit(NULL);
    return NULL;
}

// 4. 静态线程+barrier同步版本的高斯消去算法 - 按行划分
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

// 验证AVX支持
bool check_avx_support() {
    #if defined(__AVX__)
        return true;
    #else
        cout << "Warning: Compiler indicates no AVX support. Results may be unpredictable." << endl;
        return false;
    #endif
}

int main(int argc, char** argv) {
    cout << "Gaussian Elimination with AVX (Row-Based) Implementation" << endl;
    cout << "Current Date and Time (UTC): 2025-05-22 00:55:32" << endl;
    cout << "Current User: KKKyriejiang" << endl;
    
    // 检查AVX支持
    if (!check_avx_support()) {
        cout << "This program requires AVX support to run properly." << endl;
        cout << "If running on a modern x86 CPU, recompile with -mavx flag." << endl;
        cout << "Continuing anyway, but results may be incorrect." << endl;
    }
    
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
    
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Thread count: " << NUM_THREADS << endl;
    cout << "Using AVX vectorization (8 floats per operation)" << endl;
    
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
    
    // 运行并测试版本1：动态线程版本 - 按行划分
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationDynamicThread();
    gettimeofday(&end_time, NULL);
    long long dynamic_thread_time = get_execution_time();
    
    cout << "Dynamic Thread version (Row-Based) execution time: " << dynamic_thread_time << " us" << endl;
    cout << "Dynamic Thread version (Row-Based) speedup: " << (float)serial_time / dynamic_thread_time << endl;
    cout << "Dynamic Thread version (Row-Based) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本2：静态线程+信号量同步版本 - 按行划分
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticSemaphore();
    gettimeofday(&end_time, NULL);
    long long static_semaphore_time = get_execution_time();
    
    cout << "Static Semaphore version (Row-Based) execution time: " << static_semaphore_time << " us" << endl;
    cout << "Static Semaphore version (Row-Based) speedup: " << (float)serial_time / static_semaphore_time << endl;
    cout << "Static Semaphore version (Row-Based) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本3：静态线程+信号量同步+三重循环全部纳入线程函数版本 - 按行划分
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticFull();
    gettimeofday(&end_time, NULL);
    long long static_full_time = get_execution_time();
    
    cout << "Static Full Thread version (Row-Based) execution time: " << static_full_time << " us" << endl;
    cout << "Static Full Thread version (Row-Based) speedup: " << (float)serial_time / static_full_time << endl;
    cout << "Static Full Thread version (Row-Based) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本4：静态线程+barrier同步版本 - 按行划分
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationBarrier();
    gettimeofday(&end_time, NULL);
    long long barrier_time = get_execution_time();
    
    cout << "Barrier version (Row-Based) execution time: " << barrier_time << " us" << endl;
    cout << "Barrier version (Row-Based) speedup: " << (float)serial_time / barrier_time << endl;
    cout << "Barrier version (Row-Based) correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
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