#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cmath>
// 替换ARM NEON头文件为x86的SSE头文件
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
using namespace std;

// 矩阵大小
int n = 0;
// 线程数量 - 改为变量，可配置
int NUM_THREADS = 4;
// 矩阵数据
float **matrix = nullptr;

// 用于计时
struct timeval start_time, end_time;
long long execution_time;

// 同步所需的信号量和屏障
sem_t sem_main;
sem_t* sem_workerstart = nullptr;
sem_t* sem_workerend = nullptr;
sem_t sem_leader;
sem_t* sem_division = nullptr;
sem_t* sem_elimination = nullptr;
pthread_barrier_t barrier_division;
pthread_barrier_t barrier_elimination;

// 线程参数结构体
typedef struct {
    int k;       // 当前消去的行
    int t_id;    // 线程ID
} threadParam_t;

// 初始化信号量和屏障
void init_sync_primitives() {
    // 释放之前可能存在的资源
    if (sem_workerstart != nullptr) {
        delete[] sem_workerstart;
        delete[] sem_workerend;
        delete[] sem_division;
        delete[] sem_elimination;
    }
    
    // 根据实际线程数分配资源
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
    
    for (int i = 0; i < NUM_THREADS-1; i++) {
        sem_init(&sem_division[i], 0, 0);
        sem_init(&sem_elimination[i], 0, 0);
    }
    
    // 初始化屏障
    pthread_barrier_init(&barrier_division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_elimination, NULL, NUM_THREADS);
}

// 清理同步原语资源
void cleanup_sync_primitives() {
    // 销毁信号量
    sem_destroy(&sem_main);
    sem_destroy(&sem_leader);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
    
    for (int i = 0; i < NUM_THREADS-1; i++) {
        sem_destroy(&sem_division[i]);
        sem_destroy(&sem_elimination[i]);
    }
    
    // 销毁屏障
    pthread_barrier_destroy(&barrier_division);
    pthread_barrier_destroy(&barrier_elimination);
    
    // 释放内存
    delete[] sem_workerstart;
    delete[] sem_workerend;
    delete[] sem_division;
    delete[] sem_elimination;
    
    sem_workerstart = nullptr;
    sem_workerend = nullptr;
    sem_division = nullptr;
    sem_elimination = nullptr;
}

// 初始化矩阵，使用随机数填充
void init_matrix() {
    matrix = new float*[n];
    for (int i = 0; i < n; i++) {
        // 16字节对齐以优化SSE指令的访问
        matrix[i] = (float*)_mm_malloc(n * sizeof(float), 16);
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
        _mm_free(matrix[i]);
    }
    delete[] matrix;
}

// SSE向量化的行归一化函数
void division_sse(int k) {
    __m128 vt = _mm_set1_ps(matrix[k][k]); // 把除数加载到所有元素
    int j = k + 1;
    
    // 使用SSE指令进行向量化除法，一次处理4个float元素
    for (; j + 3 < n; j += 4) {
        // 加载4个元素
        __m128 va = _mm_loadu_ps(&matrix[k][j]);
        // 执行除法
        va = _mm_div_ps(va, vt);
        // 存回内存
        _mm_storeu_ps(&matrix[k][j], va);
    }
    
    // 处理剩余的元素
    for (; j < n; j++) {
        matrix[k][j] = matrix[k][j] / matrix[k][k];
    }
    
    matrix[k][k] = 1.0;
}

// SSE向量化的消去函数
void elimination_sse(int k, int i) {
    __m128 vaik = _mm_set1_ps(matrix[i][k]); // 把要减去的行首元素加载到所有元素
    int j = k + 1;
    
    // 使用SSE指令进行向量化消去，一次处理4个float元素
    for (; j + 3 < n; j += 4) {
        // 加载4个元素
        __m128 vakj = _mm_loadu_ps(&matrix[k][j]);
        __m128 vaij = _mm_loadu_ps(&matrix[i][j]);
        // 计算matrix[i][k] * matrix[k][j]
        __m128 vx = _mm_mul_ps(vaik, vakj);
        // 计算matrix[i][j] - matrix[i][k] * matrix[k][j]
        vaij = _mm_sub_ps(vaij, vx);
        // 存回内存
        _mm_storeu_ps(&matrix[i][j], vaij);
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
        division_sse(k);
        
        // 消去后续各行
        for (int i = k + 1; i < n; i++) {
            elimination_sse(k, i);
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
    elimination_sse(k, i);
    
    pthread_exit(NULL);
}

// 1. 动态线程版本的高斯消去算法
void gaussEliminationDynamicThread() {
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        division_sse(k);
        
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
        
        // 使用SSE优化的消去操作
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            elimination_sse(k, i);
        }
        
        sem_post(&sem_main); // 通知主线程已完成
        sem_wait(&sem_workerend[t_id]); // 等待所有工作线程完成当前轮次
    }
    
    pthread_exit(NULL);
}

// 2. 静态线程+信号量同步版本的高斯消去算法
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
        division_sse(k);
        
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

// 3. 静态线程+信号量同步+三重循环全部纳入线程函数版本的线程函数
void* staticFullThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // 线程0负责归一化操作
        if (t_id == 0) {
            division_sse(k);
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
            elimination_sse(k, i);
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

// 4. 静态线程+barrier同步版本的线程函数
void* staticBarrierThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // 线程0负责归一化操作
        if (t_id == 0) {
            division_sse(k);
        }
        
        // 使用屏障同步，确保归一化完成
        pthread_barrier_wait(&barrier_division);
        
        // 所有线程执行消去操作
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            elimination_sse(k, i);
        }
        
        // 使用屏障同步，确保所有消去操作完成
        pthread_barrier_wait(&barrier_elimination);
    }
    
    pthread_exit(NULL);
}

// 4. 静态线程+barrier同步版本的高斯消去算法
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
        temp_matrix[i] = (float*)_mm_malloc(n * sizeof(float), 16);
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
        _mm_free(temp_matrix[i]);
    }
    delete[] temp_matrix;
    
    return correct;
}

// 保存结果
float** save_result() {
    float** result = new float*[n];
    for (int i = 0; i < n; i++) {
        result[i] = (float*)_mm_malloc(n * sizeof(float), 16);
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
        _mm_free(result[i]);
    }
    delete[] result;
}

// 计算执行时间
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
        NUM_THREADS = atoi(argv[2]);
        if (NUM_THREADS <= 0) {
            cout << "Invalid thread count, using default (4)" << endl;
            NUM_THREADS = 4;
        }
    }
    
    cout << "Matrix size: " << n << " x " << n << endl;
    cout << "Thread count: " << NUM_THREADS << endl;
    
    // 初始化随机数生成器
    srand(42);
    
    // 初始化矩阵
    init_matrix();
    
    // 初始化同步原语
    init_sync_primitives();
    
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
    cout << "\nCSV Format for plotting:\n";
    cout << "matrix_size,thread_count,serial,dynamic_thread,static_semaphore,static_full,barrier\n";
    cout << n << "," << NUM_THREADS << "," << serial_time << "," << dynamic_thread_time << "," 
         << static_semaphore_time << "," << static_full_time << "," << barrier_time << endl;
    
    // 输出CSV格式的加速比
    cout << "\nSpeedup CSV Format for plotting:\n";
    cout << "matrix_size,thread_count,dynamic_thread,static_semaphore,static_full,barrier\n";
    cout << n << "," << NUM_THREADS << "," << (float)serial_time / dynamic_thread_time << "," 
         << (float)serial_time / static_semaphore_time << "," 
         << (float)serial_time / static_full_time << "," 
         << (float)serial_time / barrier_time << endl;
    
    // 清理同步原语
    cleanup_sync_primitives();
    
    // 释放内存
    free_result(original_matrix);
    free_result(serial_result);
    free_matrix();
    
    return 0;
}