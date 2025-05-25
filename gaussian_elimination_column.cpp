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

// NEON向量化的行归一化函数（按列划分）
void division_neon_col(int k, int t_id, int cols_per_thread) {
    // 计算线程负责的列范围
    int col_start = k + 1 + t_id * cols_per_thread;
    int col_end = col_start + cols_per_thread;
    if (col_end > n) col_end = n;
    if (col_start >= n) return;

    float pivot = matrix[k][k];
    
    // 使用NEON指令进行向量化除法
    float32x4_t vt = vdupq_n_f32(pivot);
    int j = col_start;
    
    for (; j + 3 < col_end; j += 4) {
        float32x4_t va = vld1q_f32(&matrix[k][j]);
        va = vdivq_f32(va, vt);
        vst1q_f32(&matrix[k][j], va);
    }
    
    // 处理剩余的元素
    for (; j < col_end; j++) {
        matrix[k][j] = matrix[k][j] / pivot;
    }
    
    // 由于是按列划分，只有一个线程会设置对角线元素为1
    if (k == 0 && t_id == 0) {
        matrix[k][k] = 1.0;
    }
}

// NEON向量化的消去函数（按列划分）
void elimination_neon_col(int k, int i, int t_id, int cols_per_thread) {
    // 计算线程负责的列范围
    int col_start = k + 1 + t_id * cols_per_thread;
    int col_end = col_start + cols_per_thread;
    if (col_end > n) col_end = n;
    if (col_start >= n) return;

    float multiplier = matrix[i][k];
    float32x4_t vaik = vdupq_n_f32(multiplier);
    int j = col_start;
    
    // 使用NEON指令进行向量化消去
    for (; j + 3 < col_end; j += 4) {
        float32x4_t vakj = vld1q_f32(&matrix[k][j]);
        float32x4_t vaij = vld1q_f32(&matrix[i][j]);
        float32x4_t vx = vmulq_f32(vaik, vakj);
        vaij = vsubq_f32(vaij, vx);
        vst1q_f32(&matrix[i][j], vaij);
    }
    
    // 处理剩余的元素
    for (; j < col_end; j++) {
        matrix[i][j] = matrix[i][j] - multiplier * matrix[k][j];
    }
    
    // 由于是按列划分，每个线程都要单独设置对消元位置为0
    if (t_id == 0) {
        matrix[i][k] = 0;
    }
}

// 串行高斯消去算法
void gaussEliminationSerial() {
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        float pivot = matrix[k][k];
        for (int j = k + 1; j < n; j++) {
            matrix[k][j] /= pivot;
        }
        matrix[k][k] = 1.0;
        
        // 消去后续各行
        for (int i = k + 1; i < n; i++) {
            float multiplier = matrix[i][k];
            for (int j = k + 1; j < n; j++) {
                matrix[i][j] -= multiplier * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 1. 动态线程版本的除法线程函数
void* dynamicDivisionThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    
    // 计算每个线程需要处理的列数
    int cols_per_thread = (n - k - 1) / NUM_THREADS + 1;
    
    // 执行列划分的归一化
    division_neon_col(k, t_id, cols_per_thread);
    
    pthread_exit(NULL);
}

// 1. 动态线程版本的消去线程函数
void* dynamicEliminationThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int i = p->t_id / NUM_THREADS + k + 1;  // 行索引
    int t_id = p->t_id % NUM_THREADS;       // 列划分的线程ID
    
    // 计算每个线程需要处理的列数
    int cols_per_thread = (n - k - 1) / NUM_THREADS + 1;
    
    // 执行列划分的消去
    elimination_neon_col(k, i, t_id, cols_per_thread);
    
    pthread_exit(NULL);
}

// 1. 动态线程版本的高斯消去算法（列划分）
void gaussEliminationDynamicThread() {
    for (int k = 0; k < n; k++) {
        // 使用动态线程进行归一化操作
        pthread_t* division_handles = new pthread_t[NUM_THREADS];
        threadParam_t* division_param = new threadParam_t[NUM_THREADS];
        
        // 为每个线程分配任务（列划分）
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            division_param[t_id].k = k;
            division_param[t_id].t_id = t_id;
            pthread_create(&division_handles[t_id], NULL, dynamicDivisionThreadFunc, &division_param[t_id]);
        }
        
        // 等待所有归一化线程完成
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            pthread_join(division_handles[t_id], NULL);
        }
        
        // 确保归一化的对角线元素为1
        matrix[k][k] = 1.0;
        
        // 创建消去线程
        int elimination_thread_count = (n - k - 1) * NUM_THREADS;
        if (elimination_thread_count <= 0) {
            delete[] division_handles;
            delete[] division_param;
            continue;
        }
        
        pthread_t* elimination_handles = new pthread_t[elimination_thread_count];
        threadParam_t* elimination_param = new threadParam_t[elimination_thread_count];
        
        // 为每个线程分配任务（行和列的二维划分）
        int thread_idx = 0;
        for (int i = k + 1; i < n; i++) {
            for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
                elimination_param[thread_idx].k = k;
                elimination_param[thread_idx].t_id = (i - k - 1) * NUM_THREADS + t_id;  // 组合行和列的ID
                pthread_create(&elimination_handles[thread_idx], NULL, dynamicEliminationThreadFunc, &elimination_param[thread_idx]);
                thread_idx++;
            }
        }
        
        // 等待所有消去线程完成
        for (int t_id = 0; t_id < elimination_thread_count; t_id++) {
            pthread_join(elimination_handles[t_id], NULL);
        }
        
        // 释放资源
        delete[] division_handles;
        delete[] division_param;
        delete[] elimination_handles;
        delete[] elimination_param;
    }
}

// 2. 静态线程+信号量同步版本的线程函数（列划分）
void* staticSemaphoreThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]); // 等待主线程通知开始工作
        
        // 计算每个线程需要处理的列数
        int cols_per_thread = (n - k - 1) / NUM_THREADS + 1;
        
        // 先进行归一化操作（每个线程处理不同的列）
        division_neon_col(k, t_id, cols_per_thread);
        
        sem_post(&sem_main); // 通知主线程已完成归一化
        sem_wait(&sem_workerend[t_id]); // 等待所有工作线程完成归一化
        
        // 进行消去操作（每个线程处理所有行的对应列部分）
        for (int i = k + 1; i < n; i++) {
            elimination_neon_col(k, i, t_id, cols_per_thread);
        }
        
        sem_post(&sem_main); // 通知主线程已完成消去
        sem_wait(&sem_workerend[t_id]); // 等待进入下一轮
    }
    
    pthread_exit(NULL);
}

// 2. 静态线程+信号量同步版本的高斯消去算法（列划分）
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
        // 唤醒工作线程进行归一化
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workerstart[t_id]);
        }
        
        // 等待所有工作线程完成归一化
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        
        // 确保归一化的对角线元素为1
        matrix[k][k] = 1.0;
        
        // 通知工作线程进行消去
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workerend[t_id]);
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

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的线程函数（列划分）
void* staticFullThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // 计算每个线程需要处理的列数
        int cols_per_thread = (n - k - 1) / NUM_THREADS + 1;
        
        // 线程0负责处理归一化前的同步
        if (t_id == 0) {
            // 不需要额外操作，直接进入归一化
        } else {
            sem_wait(&sem_division[t_id-1]); // 非0线程等待同步
        }
        
        // 执行归一化（每个线程处理不同的列）
        division_neon_col(k, t_id, cols_per_thread);
        
        // 确保对角线元素为1（只在一个线程中设置，避免竞争）
        if (t_id == 0) {
            matrix[k][k] = 1.0;
        }
        
        // 线程0通知其他线程归一化已完成
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_division[i]);
            }
        }
        
        // 所有线程执行消去操作（每个线程处理所有行的对应列部分）
        for (int i = k + 1; i < n; i++) {
            elimination_neon_col(k, i, t_id, cols_per_thread);
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

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的高斯消去算法（列划分）
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

// 4. 静态线程+barrier同步版本的线程函数（列划分）
void* staticBarrierThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // 计算每个线程需要处理的列数
        int cols_per_thread = (n - k - 1) / NUM_THREADS + 1;
        
        // 执行归一化（每个线程处理不同的列）
        division_neon_col(k, t_id, cols_per_thread);
        
        // 确保对角线元素为1（只在一个线程中设置，避免竞争）
        if (t_id == 0) {
            matrix[k][k] = 1.0;
        }
        
        // 使用屏障同步，确保归一化完成
        pthread_barrier_wait(&barrier_division);
        
        // 所有线程执行消去操作（每个线程处理所有行的对应列部分）
        for (int i = k + 1; i < n; i++) {
            elimination_neon_col(k, i, t_id, cols_per_thread);
        }
        
        // 使用屏障同步，确保所有消去操作完成
        pthread_barrier_wait(&barrier_elimination);
    }
    
    pthread_exit(NULL);
}

// 4. 静态线程+barrier同步版本的高斯消去算法（列划分）
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

int main(int argc, char** argv) {

    // 设置线程数（如果指定）
    if (argc >= 3) {
        NUM_THREADS = atoi(argv[2]);
        cout << "Using " << NUM_THREADS << " threads" << endl;
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
    
    // 运行并测试版本1：动态线程版本（列划分）
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationDynamicThread();
    gettimeofday(&end_time, NULL);
    long long dynamic_thread_time = get_execution_time();
    
    cout << "Dynamic Thread version (column division) execution time: " << dynamic_thread_time << " us" << endl;
    cout << "Dynamic Thread version speedup: " << (float)serial_time / dynamic_thread_time << endl;
    cout << "Dynamic Thread version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本2：静态线程+信号量同步版本（列划分）
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticSemaphore();
    gettimeofday(&end_time, NULL);
    long long static_semaphore_time = get_execution_time();
    
    cout << "Static Semaphore version (column division) execution time: " << static_semaphore_time << " us" << endl;
    cout << "Static Semaphore version speedup: " << (float)serial_time / static_semaphore_time << endl;
    cout << "Static Semaphore version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本3：静态线程+信号量同步+三重循环全部纳入线程函数版本（列划分）
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationStaticFull();
    gettimeofday(&end_time, NULL);
    long long static_full_time = get_execution_time();
    
    cout << "Static Full Thread version (column division) execution time: " << static_full_time << " us" << endl;
    cout << "Static Full Thread version speedup: " << (float)serial_time / static_full_time << endl;
    cout << "Static Full Thread version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本4：静态线程+barrier同步版本（列划分）
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationBarrier();
    gettimeofday(&end_time, NULL);
    long long barrier_time = get_execution_time();
    
    cout << "Barrier version (column division) execution time: " << barrier_time << " us" << endl;
    cout << "Barrier version speedup: " << (float)serial_time / barrier_time << endl;
    cout << "Barrier version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 输出CSV格式的执行时间
    cout << "\nCSV Format for plotting:\n";
    cout << "matrix_size,serial,dynamic_thread_col,static_semaphore_col,static_full_col,barrier_col\n";
    cout << n << "," << serial_time << "," << dynamic_thread_time << "," 
         << static_semaphore_time << "," << static_full_time << "," << barrier_time << endl;
    
    // 输出CSV格式的加速比
    cout << "\nSpeedup CSV Format for plotting:\n";
    cout << "matrix_size,dynamic_thread_col,static_semaphore_col,static_full_col,barrier_col\n";
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