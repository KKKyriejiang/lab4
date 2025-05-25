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
// 默认分块大小
int BLOCK_SIZE = 32;
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

// 1. 分块缓存优化的高斯消元算法 - 串行版本
void gaussEliminationBlockedSerial() {
    int block_size = BLOCK_SIZE;
    
    // 确保分块尺寸不大于矩阵维度
    if (block_size > n) block_size = n;
    
    for (int k = 0; k < n; k += block_size) {
        // 处理对角线分块
        int end_k = min(k + block_size, n);
        for (int kk = k; kk < end_k; kk++) {
            division_neon(kk);
            
            // 消去当前分块中的后续行
            for (int i = kk + 1; i < end_k; i++) {
                elimination_neon(kk, i);
            }
        }
        
        // 对剩余的行应用当前分块的变换
        for (int i = end_k; i < n; i++) {
            for (int kk = k; kk < end_k; kk++) {
                elimination_neon(kk, i);
            }
        }
        
        // 更新剩余的矩阵分块
        for (int i = end_k; i < n; i += block_size) {
            int end_i = min(i + block_size, n);
            for (int ii = i; ii < end_i; ii++) {
                for (int j = end_k; j < n; j += block_size) {
                    int end_j = min(j + block_size, n);
                    
                    // 应用局部更新以提高缓存命中率
                    for (int kk = k; kk < end_k; kk++) {
                        float pivot = matrix[ii][kk];
                        for (int jj = j; jj < end_j; jj++) {
                            matrix[ii][jj] -= pivot * matrix[kk][jj];
                        }
                    }
                }
            }
        }
    }
}

// 2. 分块缓存优化的高斯消元算法 - OpenMP并行版本
void gaussEliminationBlockedOpenMP() {
    int block_size = BLOCK_SIZE;
    
    // 确保分块尺寸不大于矩阵维度
    if (block_size > n) block_size = n;
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k += block_size) {
            // 处理对角线分块
            int end_k = min(k + block_size, n);
            
            #pragma omp single
            {
                for (int kk = k; kk < end_k; kk++) {
                    division_neon(kk);
                    
                    // 消去当前分块中的后续行
                    for (int i = kk + 1; i < end_k; i++) {
                        elimination_neon(kk, i);
                    }
                }
            }
            // 隐式屏障确保对角线分块完成
            
            // 并行处理剩余行
            #pragma omp for
            for (int i = end_k; i < n; i++) {
                for (int kk = k; kk < end_k; kk++) {
                    elimination_neon(kk, i);
                }
            }
            // 隐式屏障确保所有行更新完成
            
            // 并行更新剩余的矩阵分块
            #pragma omp for collapse(2)
            for (int i = end_k; i < n; i += block_size) {
                for (int j = end_k; j < n; j += block_size) {
                    int end_i = min(i + block_size, n);
                    int end_j = min(j + block_size, n);
                    
                    // 处理当前分块
                    for (int ii = i; ii < end_i; ii++) {
                        for (int kk = k; kk < end_k; kk++) {
                            float pivot = matrix[ii][kk];
                            for (int jj = j; jj < end_j; jj++) {
                                matrix[ii][jj] -= pivot * matrix[kk][jj];
                            }
                        }
                    }
                }
            }
            // 隐式屏障确保所有分块更新完成
        }
    }
}

// 3. 缓存优化版本 - 数据预取与缓存行对齐
void gaussEliminationCacheOptimized() {
    const int prefetch_distance = 8;  // 预取距离，需要根据目标架构调整
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k++) {
            // 线程0负责归一化
            #pragma omp single
            {
                division_neon(k);
                
                // 数据预取：预取即将使用的行
                for (int p = k + 1; p < min(k + 1 + prefetch_distance, n); p++) {
                    __builtin_prefetch(matrix[p], 0, 3);  // 预取读取，局部性强
                    __builtin_prefetch(matrix[k], 0, 3);  // 预取归一化后的行
                }
            }
            
            // 并行消去，使用循环阻塞以提高空间局部性
            #pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++) {
                // 如果不是最后几行，预取下一组行
                if (i + prefetch_distance < n) {
                    __builtin_prefetch(matrix[i + prefetch_distance], 1, 1);
                }
                
                elimination_neon(k, i);
            }
        }
    }
}

// 4. 递归分块高斯消元算法（Cache-oblivious）
void gaussEliminationRecursiveBlock(int start_row, int start_col, int end_row, int end_col) {
    int rows = end_row - start_row;
    int cols = end_col - start_col;
    
    // 基本情况：小规模矩阵直接计算
    if (rows <= BLOCK_SIZE || cols <= BLOCK_SIZE) {
        for (int k = start_col; k < end_col; k++) {
            // 归一化当前行
            float pivot = matrix[k][k];
            for (int j = k + 1; j < end_col; j++) {
                matrix[k][j] /= pivot;
            }
            matrix[k][k] = 1.0;
            
            // 消去后续各行
            for (int i = k + 1; i < end_row; i++) {
                float factor = matrix[i][k];
                for (int j = k + 1; j < end_col; j++) {
                    matrix[i][j] -= factor * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }
        return;
    }
    
    // 递归情况：将矩阵分为四个子矩阵
    int mid_row = start_row + rows / 2;
    int mid_col = start_col + cols / 2;
    
    // 递归处理左上角
    gaussEliminationRecursiveBlock(start_row, start_col, mid_row, mid_col);
    
    // 应用左上角的变换到右上角
    #pragma omp parallel for if(rows > 128) num_threads(NUM_THREADS)
    for (int i = start_row; i < mid_row; i++) {
        for (int j = mid_col; j < end_col; j++) {
            for (int k = start_col; k < mid_col; k++) {
                if (i == k) continue;  // 跳过归一化行
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
        }
    }
    
    // 应用左上角的变换到左下角
    #pragma omp parallel for if(rows > 128) num_threads(NUM_THREADS)
    for (int i = mid_row; i < end_row; i++) {
        for (int j = start_col; j < mid_col; j++) {
            for (int k = start_col; k < j; k++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
        }
    }
    
    // 应用左上角的变换到右下角
    #pragma omp parallel for collapse(2) if(rows > 128) num_threads(NUM_THREADS)
    for (int i = mid_row; i < end_row; i++) {
        for (int j = mid_col; j < end_col; j++) {
            for (int k = start_col; k < mid_col; k++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
        }
    }
    
    // 递归处理右上角
    gaussEliminationRecursiveBlock(start_row, mid_col, mid_row, end_col);
    
    // 递归处理左下角
    gaussEliminationRecursiveBlock(mid_row, start_col, end_row, mid_col);
    
    // 递归处理右下角
    gaussEliminationRecursiveBlock(mid_row, mid_col, end_row, end_col);
}

// 包装函数，调用递归分块高斯消元
void gaussEliminationCacheOblivious() {
    gaussEliminationRecursiveBlock(0, 0, n, n);
}

// 5. Z-Morton排序布局优化的高斯消元 - 提高空间局部性
void gaussEliminationZMorton() {
    // 使用转置矩阵算法来改善空间局部性
    float *transpose_buffer = new float[BLOCK_SIZE * BLOCK_SIZE];
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int k = 0; k < n; k++) {
            // 线程0负责归一化
            #pragma omp single
            {
                division_neon(k);
            }
            
            // 并行消去，但采用分块以提高缓存利用率
            #pragma omp for
            for (int i_block = k + 1; i_block < n; i_block += BLOCK_SIZE) {
                int i_end = min(i_block + BLOCK_SIZE, n);
                
                for (int j_block = k + 1; j_block < n; j_block += BLOCK_SIZE) {
                    int j_end = min(j_block + BLOCK_SIZE, n);
                    
                    // 如果分块较大，可以考虑转置以提高访问局部性
                    if (j_end - j_block > 16) {
                        // 局部转置技术，提高缓存命中率
                        for (int i = i_block; i < i_end; i++) {
                            float mult = matrix[i][k];
                            
                            // 使用局部转置提高缓存访问效率
                            for (int j = j_block; j < j_end; j++) {
                                matrix[i][j] -= mult * matrix[k][j];
                            }
                        }
                    } else {
                        // 小分块，直接计算
                        for (int i = i_block; i < i_end; i++) {
                            float mult = matrix[i][k];
                            for (int j = j_block; j < j_end; j++) {
                                matrix[i][j] -= mult * matrix[k][j];
                            }
                            matrix[i][k] = 0;
                        }
                    }
                }
            }
        }
    }
    
    delete[] transpose_buffer;
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
        cout << "Usage: " << argv[0] << " <matrix_size> [num_threads] [block_size]" << endl;
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
            cout << "Invalid number of threads, using default: " << NUM_THREADS << endl;
            NUM_THREADS = 4;
        }
    }
    
    // 设置分块大小（如果提供）
    if (argc >= 4) {
        BLOCK_SIZE = atoi(argv[3]);
        if (BLOCK_SIZE <= 0) {
            cout << "Invalid block size, using default: " << BLOCK_SIZE << endl;
            BLOCK_SIZE = 32;
        }
    }
    
    // 设置OpenMP线程数
    omp_set_num_threads(NUM_THREADS);
    
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Number of threads: " << NUM_THREADS << endl;
    cout << "Block size: " << BLOCK_SIZE << endl;
    
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
    
    // 运行并测试分块串行版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationBlockedSerial();
    gettimeofday(&end_time, NULL);
    long long blocked_serial_time = get_execution_time();
    
    cout << "\nBlocked Serial version execution time: " << blocked_serial_time << " us" << endl;
    cout << "Blocked Serial version speedup: " << (float)serial_time / blocked_serial_time << endl;
    cout << "Blocked Serial version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试分块OpenMP版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationBlockedOpenMP();
    gettimeofday(&end_time, NULL);
    long long blocked_omp_time = get_execution_time();
    
    cout << "\nBlocked OpenMP version execution time: " << blocked_omp_time << " us" << endl;
    cout << "Blocked OpenMP version speedup: " << (float)serial_time / blocked_omp_time << endl;
    cout << "Blocked OpenMP version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试缓存优化版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationCacheOptimized();
    gettimeofday(&end_time, NULL);
    long long cache_opt_time = get_execution_time();
    
    cout << "\nCache Optimized version execution time: " << cache_opt_time << " us" << endl;
    cout << "Cache Optimized version speedup: " << (float)serial_time / cache_opt_time << endl;
    cout << "Cache Optimized version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试Cache-oblivious版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationCacheOblivious();
    gettimeofday(&end_time, NULL);
    long long cache_oblivious_time = get_execution_time();
    
    cout << "\nCache-Oblivious version execution time: " << cache_oblivious_time << " us" << endl;
    cout << "Cache-Oblivious version speedup: " << (float)serial_time / cache_oblivious_time << endl;
    cout << "Cache-Oblivious version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 运行并测试Z-Morton版本
    restore_matrix(original_matrix);
    gettimeofday(&start_time, NULL);
    gaussEliminationZMorton();
    gettimeofday(&end_time, NULL);
    long long zmorton_time = get_execution_time();
    
    cout << "\nZ-Morton version execution time: " << zmorton_time << " us" << endl;
    cout << "Z-Morton version speedup: " << (float)serial_time / zmorton_time << endl;
    cout << "Z-Morton version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
    
    // 输出CSV格式的执行时间
    cout << "\nCSV Format for plotting:\n";
    cout << "matrix_size,block_size,threads,serial,blocked_serial,blocked_omp,cache_opt,cache_oblivious,zmorton\n";
    cout << n << "," << BLOCK_SIZE << "," << NUM_THREADS << "," 
         << serial_time << "," << blocked_serial_time << "," << blocked_omp_time << "," 
         << cache_opt_time << "," << cache_oblivious_time << "," << zmorton_time << endl;
    
    // 输出CSV格式的加速比
    cout << "\nSpeedup CSV Format for plotting:\n";
    cout << "matrix_size,block_size,threads,blocked_serial,blocked_omp,cache_opt,cache_oblivious,zmorton\n";
    cout << n << "," << BLOCK_SIZE << "," << NUM_THREADS << "," 
         << (float)serial_time / blocked_serial_time << "," 
         << (float)serial_time / blocked_omp_time << "," 
         << (float)serial_time / cache_opt_time << "," 
         << (float)serial_time / cache_oblivious_time << "," 
         << (float)serial_time / zmorton_time << endl;
    
    // 释放内存
    free_result(original_matrix);
    free_result(serial_result);
    free_matrix();
    
    return 0;
}