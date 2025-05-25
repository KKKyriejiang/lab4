#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#include <sys/time.h>
#include <arm_neon.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include <random>

using namespace std;

// 矩阵参数定义
// 选择要使用的矩阵规模
#define MATRIX_SMALL 1
#define MATRIX_MEDIUM 0
#define MATRIX_LARGE 0
#define MATRIX_XLARGE 0

#if MATRIX_SMALL
    const int lieNum = 1000;    // 总列数
    const int Num = 32;        // 每行占用的unsigned int数量 (列数/32 + 1)
    const int pasNum = 500;    // 被消元行数量
#elif MATRIX_MEDIUM
    const int lieNum = 2000;   // 总列数
    const int Num = 63;        // 每行占用的unsigned int数量
    const int pasNum = 1000;   // 被消元行数量
#elif MATRIX_LARGE
    const int lieNum = 4000;   // 总列数
    const int Num = 125;       // 每行占用的unsigned int数量
    const int pasNum = 2000;   // 被消元行数量
#elif MATRIX_XLARGE
    const int lieNum = 8000;   // 总列数
    const int Num = 250;       // 每行占用的unsigned int数量
    const int pasNum = 4000;   // 被消元行数量
#else
    // 默认小规模
    const int lieNum = 1000;
    const int Num = 32;
    const int pasNum = 500;
#endif

// 消元子和被消元行的位向量表示
unsigned int Act[lieNum][Num+1] = {0};
unsigned int Pas[pasNum][Num+1] = {0};

// 线程数量
int NUM_THREADS = 4;

// 互斥锁，用于保护升格操作
omp_lock_t upgrade_lock;

// 调试标志
bool DEBUG = false;

// 打印特殊矩阵的函数（用于调试）
void printMatrix(unsigned int mat[][Num+1], int rows, int max_rows = 10, int max_cols = 5) {
    if (!DEBUG) return;
    
    cout << "Matrix (" << rows << " x " << Num << "):" << endl;
    for (int i = 0; i < min(rows, max_rows); i++) {
        cout << "Row " << i << ": ";
        for (int j = 0; j < min(Num, max_cols); j++) {
            cout << hex << mat[i][j] << " ";
        }
        cout << dec << (max_cols < Num ? "..." : "") << endl;
    }
    if (rows > max_rows) cout << "..." << endl;
}

// 生成随机测试数据
void generate_test_data() {
    cout << "正在生成随机测试数据..." << endl;
    
    // 使用固定的种子以确保可重现性
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, lieNum-1);
    
    // 为80%的列生成消元子
    int num_active_cols = lieNum * 0.8;
    for (int i = 0; i < num_active_cols; i++) {
        int col = dist(rng) % lieNum;
        
        // 设置消元子有效标志
        Act[col][Num] = 1;
        
        // 设置对角线元素（首项）
        int uint_pos = col / 32;
        int bit_pos = col % 32;
        Act[col][Num - 1 - uint_pos] |= (1u << bit_pos);
        
        // 随机设置约10%的其他位
        int num_bits = Num * 32 * 0.1;
        for (int j = 0; j < num_bits; j++) {
            int random_col = dist(rng);
            uint_pos = random_col / 32;
            bit_pos = random_col % 32;
            Act[col][Num - 1 - uint_pos] |= (1u << bit_pos);
        }
    }
    
    // 生成被消元行
    for (int i = 0; i < pasNum; i++) {
        // 随机选择约10%的位设置为1
        int num_bits = Num * 32 * 0.1;
        for (int j = 0; j < num_bits; j++) {
            int random_col = dist(rng);
            int uint_pos = random_col / 32;
            int bit_pos = random_col % 32;
            Pas[i][Num - 1 - uint_pos] |= (1u << bit_pos);
        }
        
        // 查找首项位置
        int first_one = -1;
        for (int j = 0; j < Num; j++) {
            if (Pas[i][j] != 0) {
                unsigned int temp = Pas[i][j];
                int pos = j * 32;
                while (temp != 0) {
                    if (temp & 1) {
                        first_one = pos;
                        break;
                    }
                    temp >>= 1;
                    pos++;
                }
                break;
            }
        }
        
        // 设置首项位置
        Pas[i][Num] = first_one;
    }
    
    cout << "随机测试数据生成完成。" << endl;
}

// 从文件初始化消元子
bool init_A(const string& filename = "act3.txt") {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cout << "无法打开消元子文件: " << filename << "，将使用随机生成的数据。" << endl;
        return false;
    }
    
    cout << "正在加载消元子..." << endl;
    
    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        unsigned int a;
        int index = -1;
        bool first = true;
        
        while (ss >> a) {
            if (first) {
                // 取每行第一个数字为行标（消元子首项）
                index = a;
                first = false;
            }
            
            // 将对应位设置为1
            int bit_pos = a % 32;
            int uint_pos = a / 32;
            if (uint_pos < Num - 1) {
                Act[index][Num - 1 - uint_pos] |= (1u << bit_pos);
            }
        }
        
        if (index >= 0 && index < lieNum) {
            // 设置该行的有效标志
            Act[index][Num] = 1;
        }
    }
    
    infile.close();
    cout << "消元子加载完成。" << endl;
    
    if (DEBUG) {
        printMatrix(Act, lieNum);
    }
    return true;
}

// 从文件初始化被消元行
bool init_P(const string& filename = "pas3.txt") {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cout << "无法打开被消元行文件: " << filename << "，将使用随机生成的数据。" << endl;
        return false;
    }
    
    cout << "正在加载被消元行..." << endl;
    
    string line;
    int row_index = 0;
    
    while (getline(infile, line) && row_index < pasNum) {
        stringstream ss(line);
        unsigned int a;
        bool first = true;
        
        while (ss >> a) {
            if (first) {
                // 存储每行的首项（用于消元操作）
                Pas[row_index][Num] = a;
                first = false;
            }
            
            // 将对应位设置为1
            int bit_pos = a % 32;
            int uint_pos = a / 32;
            if (uint_pos < Num - 1) {
                Pas[row_index][Num - 1 - uint_pos] |= (1u << bit_pos);
            }
        }
        
        row_index++;
    }
    
    infile.close();
    cout << "被消元行加载完成，共加载 " << row_index << " 行。" << endl;
    
    if (DEBUG) {
        printMatrix(Pas, pasNum);
    }
    return true;
}

// 查找一行中的第一个非零位的位置
int findFirstOne(unsigned int row[Num+1]) {
    for (int i = 0; i < Num; i++) {
        if (row[i] != 0) {
            unsigned int temp = row[i];
            int pos = i * 32;
            
            // 找出这个uint中的第一个1的位置
            while (temp != 0) {
                if (temp & 1) {
                    return pos;
                }
                temp >>= 1;
                pos++;
            }
            return pos - 1;
        }
    }
    return -1;  // 没有非零位
}

// 计算更新被消元行首项的函数
void updateFirstOne(unsigned int row[Num+1]) {
    int pos = findFirstOne(row);
    row[Num] = pos;
}

// 使用NEON指令优化的异或操作
void xorRowsNeon(unsigned int* target, const unsigned int* source) {
    int k;
    for (k = 0; k + 3 < Num; k += 4) {
        uint32x4_t va_target = vld1q_u32(&target[k]);
        uint32x4_t va_source = vld1q_u32(&source[k]);
        va_target = veorq_u32(va_target, va_source);
        vst1q_u32(&target[k], va_target);
    }
    
    // 处理剩余部分
    for (; k < Num; k++) {
        target[k] ^= source[k];
    }
}

// OpenMP版本的特殊高斯消元算法
void gaussEliminationOpenMP() {
    cout << "使用 " << NUM_THREADS << " 个线程执行OpenMP特殊高斯消元..." << endl;
    
    bool continue_elimination;
    
    do {
        continue_elimination = false;
        
        // 并行处理消元过程
        #pragma omp parallel num_threads(NUM_THREADS) shared(continue_elimination)
        {
            // 每个线程处理部分被消元行
            #pragma omp for schedule(dynamic, 8)
            for (int i = 0; i < pasNum; i++) {
                // 跳过已经被升格的行
                if (Pas[i][Num] == (unsigned int)(-1)) {
                    continue;
                }
                
                // 进行消元
                while (true) {
                    int pivot = Pas[i][Num];
                    if (pivot == -1) break; // 该行已被消元
                    
                    if (Act[pivot][Num] == 1) {
                        // 有对应的消元子，执行异或消元
                        xorRowsNeon(Pas[i], Act[pivot]);
                        updateFirstOne(Pas[i]);
                    } else {
                        // 没有对应的消元子，跳出当前被消元行的处理
                        break;
                    }
                }
            }
            
            // 升格操作（需要互斥保护）
            #pragma omp barrier
            
            #pragma omp for
            for (int i = 0; i < pasNum; i++) {
                if (Pas[i][Num] != (unsigned int)(-1)) {  // 未升格的行
                    int pivot = Pas[i][Num];
                    
                    if (pivot >= 0 && pivot < lieNum) {
                        #pragma omp critical
                        {
                            if (Act[pivot][Num] == 0) {  // 对应消元子为空
                                // 升格：被消元行升级为消元子
                                for (int k = 0; k < Num+1; k++) {
                                    Act[pivot][k] = Pas[i][k];
                                }
                                Act[pivot][Num] = 1;  // 设置消元子有效标志
                                Pas[i][Num] = -1;     // 标记被消元行已升格
                                continue_elimination = true;
                            }
                        }
                    }
                }
            }
        }
        
        // 如果有行被升格为消元子，则继续迭代
    } while (continue_elimination);
    
    cout << "特殊高斯消元完成。" << endl;
}

// 验证结果的正确性
bool verifyResults() {
    // 检查所有被消元行是否已被完全消元或升格
    for (int i = 0; i < pasNum; i++) {
        if (Pas[i][Num] != (unsigned int)(-1)) { 
            // 未被升格的行应该要么已经被消为0，要么其首项对应的列有消元子
            int pivot = Pas[i][Num];
            if (pivot >= 0 && pivot < lieNum && Act[pivot][Num] == 0) {
                cout << "验证失败：行 " << i << " 未被完全消元且未升格（首项位置 " << pivot << "）。" << endl;
                return false;
            }
        }
    }
    
    cout << "验证成功：所有被消元行已被适当处理。" << endl;
    return true;
}

// 保存结果到文件
void saveResults(const string& filename = "result.txt") {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "无法创建输出文件: " << filename << endl;
        return;
    }
    
    for (int i = 0; i < pasNum; i++) {
        // 对于升格的行，输出空行
        if (Pas[i][Num] == (unsigned int)(-1)) {
            outfile << endl;
            continue;
        }
        
        // 输出行中的所有1的位置
        bool first = true;
        for (int j = 0; j < Num; j++) {
            unsigned int val = Pas[i][j];
            for (int bit = 0; bit < 32; bit++) {
                if (val & (1u << bit)) {
                    int pos = j * 32 + bit;
                    if (!first) outfile << " ";
                    outfile << pos;
                    first = false;
                }
            }
        }
        outfile << endl;
    }
    
    outfile.close();
    cout << "结果已保存到 " << filename << endl;
}

// 获取当前时间的字符串表示
string getCurrentTime() {
    time_t now = time(0);
    struct tm* timeinfo = localtime(&now);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    return string(buffer);
}

// 主函数
int main(int argc, char* argv[]) {
    cout << "特殊高斯消元算法 - OpenMP版本" << endl;
    cout << "开始时间: " << getCurrentTime() << endl;
    cout << "矩阵规模: 列数 = " << lieNum << ", 被消元行数 = " << pasNum << endl;
    
    // 处理命令行参数
    if (argc > 1) {
        NUM_THREADS = atoi(argv[1]);
    }
    
    // 初始化OpenMP锁
    omp_init_lock(&upgrade_lock);
    
    // 尝试从文件加载数据，如果失败，则生成随机数据
    bool loaded_act = init_A();
    bool loaded_pas = init_P();
    
    if (!loaded_act || !loaded_pas) {
        generate_test_data();
    }
    
    // 计时
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    // 执行高斯消元
    gaussEliminationOpenMP();
    
    gettimeofday(&end_time, NULL);
    double execution_time = ((end_time.tv_sec - start_time.tv_sec) * 1000000LL + (end_time.tv_usec - start_time.tv_usec)) / 1000.0;
    
    cout << "执行时间: " << execution_time << " ms" << endl;
    
    // 验证结果
    verifyResults();
    
    // 保存结果（可选）
    // saveResults();
    
    // 释放OpenMP锁
    omp_destroy_lock(&upgrade_lock);
    
    cout << "结束时间: " << getCurrentTime() << endl;
    
    // 输出CSV格式的执行时间（用于绘图）
    cout << "\nCSV Format for plotting:\n";
    cout << "threads,rows,cols,time_ms\n";
    cout << NUM_THREADS << "," << pasNum << "," << lieNum << "," << execution_time << endl;
    
    return 0;
}