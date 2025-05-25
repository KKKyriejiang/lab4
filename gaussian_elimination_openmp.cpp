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
const int NUM_THREADS = 4;
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

// 1. OpenMP基本并行版本
void gaussEliminationBasicOpenMP() {
    for (int k = 0; k < n; k++) {
        // 归一化当前行（串行部分）
        division_neon(k);
        
        // 并行消去后续各行
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = k + 1; i < n; i++) {
            elimination_neon(k, i);
        }
    }
}

// 2. OpenMP优化版本 - 使用单一并行区域
void gaussEliminationSingleRegionOpenMP() {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k++) {
            // 线程0负责归一化
            #pragma omp single
            {
                division_neon(k);
            }
            
            // 隐式屏障同步，确保归一化完成后再进行消去
            
            // 并行消去后续各行
            #pragma omp for
            for (int i = k + 1; i < n; i++) {
                elimination_neon(k, i);
            }
            // 隐式屏障同步，确保所有消去完成
        }
    }
}

// 3. OpenMP优化版本 - 使用nowait减少同步开销
void gaussEliminationNowaitOpenMP() {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k++) {
            // 线程0负责归一化
            #pragma omp single
            {
                division_neon(k);
            }
            // 不需要显式等待，因为single指令隐含barrier
            
            // 并行消去后续各行，使用nowait避免多余的同步
            #pragma omp for nowait
            for (int i = k + 1; i < n; i++) {
                elimination_neon(k, i);
            }
            
            // 由于使用了nowait，需要显式同步以确保当前迭代完成
            #pragma omp barrier
        }
    }
}

// 4. OpenMP优化版本 - 使用调度策略优化负载均衡
void gaussEliminationScheduleOpenMP() {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k++) {
            // 线程0负责归一化
            #pragma omp single
            {
                division_neon(k);
            }
            
            // 使用dynamic调度策略优化负载均衡
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
    int num_threads = NUM_THREADS;
    if (argc >= 3) {
        num_threads = atoi(argv[2]);
        if (num_threads <= 0) {
            cout << "Invalid number of threads, using default: " << NUM_THREADS << endl;
            num_threads = NUM_THREADS;
        }
    }
    
    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Number of threads: " << num_threads << endl;
    
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
    
    cout << "\nSerial version execution time: " << serial_time << " us" << endl;
    
    // 运行并测试版本1：基本OpenMP版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationBasicOpenMP();
    gettimeofday(&end_time, NULL);
    long long basic_omp_time = get_execution_time();
    
    cout << "\nBasic OpenMP version execution time: " << basic_omp_time << " us" << endl;
    cout << "Basic OpenMP version speedup: " << (float)serial_time / basic_omp_time << endl;
    cout << "Basic OpenMP version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本2：单一并行区域OpenMP版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationSingleRegionOpenMP();
    gettimeofday(&end_time, NULL);
    long long single_region_time = get_execution_time();
    
    cout << "\nSingle Region OpenMP version execution time: " << single_region_time << " us" << endl;
    cout << "Single Region OpenMP version speedup: " << (float)serial_time / single_region_time << endl;
    cout << "Single Region OpenMP version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本3：使用nowait的OpenMP版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationNowaitOpenMP();
    gettimeofday(&end_time, NULL);
    long long nowait_time = get_execution_time();
    
    cout << "\nNowait OpenMP version execution time: " << nowait_time << " us" << endl;
    cout << "Nowait OpenMP version speedup: " << (float)serial_time / nowait_time << endl;
    cout << "Nowait OpenMP version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试版本4：使用动态调度的OpenMP版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationScheduleOpenMP();
    gettimeofday(&end_time, NULL);
    long long schedule_time = get_execution_time();
    
    cout << "\nSchedule OpenMP version execution time: " << schedule_time << " us" << endl;
    cout << "Schedule OpenMP version speedup: " << (float)serial_time / schedule_time << endl;
    cout << "Schedule OpenMP version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 输出CSV格式的执行时间
    cout << "\nCSV Format for plotting:\n";
    cout << "matrix_size,serial,basic_omp,single_region,nowait,schedule\n";
    cout << n << "," << serial_time << "," << basic_omp_time << "," 
         << single_region_time << "," << nowait_time << "," << schedule_time << endl;
    
    // 输出CSV格式的加速比
    cout << "\nSpeedup CSV Format for plotting:\n";
    cout << "matrix_size,basic_omp,single_region,nowait,schedule\n";
    cout << n << "," << (float)serial_time / basic_omp_time << "," 
         << (float)serial_time / single_region_time << "," 
         << (float)serial_time / nowait_time << "," 
         << (float)serial_time / schedule_time << endl;
    
    // 释放内存
    free_result(original_matrix);
    free_result(serial_result);
    free_matrix();
    
    return 0;
}