#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cmath>
#include <immintrin.h>  // For AVX/AVX2
using namespace std;

// Matrix size
int n = 0;
// Number of threads - changed from const to variable
int NUM_THREADS = 4;
// Matrix data
float **matrix = nullptr;

// For timing
struct timeval start_time, end_time;
long long execution_time;

// Synchronization semaphores and barriers
sem_t sem_main;
sem_t* sem_workerstart = nullptr;
sem_t* sem_workerend = nullptr;
sem_t sem_leader;
sem_t* sem_division = nullptr;
sem_t* sem_elimination = nullptr;
pthread_barrier_t barrier_division;
pthread_barrier_t barrier_elimination;

// Thread parameter struct
typedef struct {
    int k;       // Current elimination row
    int t_id;    // Thread ID
} threadParam_t;

// Initialize synchronization primitives based on thread count
void init_sync_primitives() {
    // Free any previously allocated resources
    if (sem_workerstart != nullptr) {
        delete[] sem_workerstart;
        delete[] sem_workerend; 
        delete[] sem_division;
        delete[] sem_elimination;
    }
    
    // Allocate resources based on current thread count
    sem_workerstart = new sem_t[NUM_THREADS];
    sem_workerend = new sem_t[NUM_THREADS];
    sem_division = new sem_t[NUM_THREADS-1];
    sem_elimination = new sem_t[NUM_THREADS-1];
    
    // Initialize semaphores
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
    
    // Initialize barriers
    pthread_barrier_init(&barrier_division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_elimination, NULL, NUM_THREADS);
}

// Cleanup synchronization resources
void cleanup_sync_primitives() {
    // Destroy semaphores
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
    
    // Destroy barriers
    pthread_barrier_destroy(&barrier_division);
    pthread_barrier_destroy(&barrier_elimination);
    
    // Free allocated memory
    delete[] sem_workerstart;
    delete[] sem_workerend;
    delete[] sem_division;
    delete[] sem_elimination;
    
    // Reset pointers to prevent double free
    sem_workerstart = nullptr;
    sem_workerend = nullptr;
    sem_division = nullptr;
    sem_elimination = nullptr;
}

// Safe memory allocation with error checking
void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == nullptr) {
        cout << "Memory allocation failed for size " << size << endl;
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Initialize matrix with random values
void init_matrix() {
    cout << "Initializing matrix of size " << n << "x" << n << endl;
    
    matrix = new float*[n];
    for (int i = 0; i < n; i++) {
        // Use standard malloc instead of aligned_alloc for better compatibility
        matrix[i] = new float[n + 8];  // Add padding for AVX alignment safety
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 100 + 1;
        }
    }
    // Ensure matrix is invertible
    for (int i = 0; i < n; i++) {
        matrix[i][i] += n * 10;
    }
    
    cout << "Matrix initialization complete" << endl;
}

// Free matrix memory
void free_matrix() {
    for (int i = 0; i < n; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// AVX vectorized row normalization function
void division_avx(int k) {
    // Load the pivot into all lanes of a YMM register
    float pivot = matrix[k][k];
    __m256 vt = _mm256_set1_ps(pivot);
    int j = k + 1;
    
    // Process 8 elements at a time using AVX (with unaligned loads/stores for safety)
    for (; j + 7 < n; j += 8) {
        // Load 8 elements (using unaligned load for safety)
        __m256 va = _mm256_loadu_ps(&matrix[k][j]);
        // Divide by the pivot
        va = _mm256_div_ps(va, vt);
        // Store back 8 elements (using unaligned store for safety)
        _mm256_storeu_ps(&matrix[k][j], va);
    }
    
    // Handle remaining elements
    for (; j < n; j++) {
        matrix[k][j] = matrix[k][j] / pivot;
    }
    
    matrix[k][k] = 1.0;
}

// AVX vectorized elimination function
void elimination_avx(int k, int i) {
    // Load the multiplier into all lanes
    float factor = matrix[i][k];
    __m256 vaik = _mm256_set1_ps(factor);
    int j = k + 1;
    
    // Process 8 elements at a time using AVX (with unaligned loads/stores for safety)
    for (; j + 7 < n; j += 8) {
        // Load 8 elements from pivot row (using unaligned load for safety)
        __m256 vakj = _mm256_loadu_ps(&matrix[k][j]);
        // Load 8 elements from target row (using unaligned load for safety)
        __m256 vaij = _mm256_loadu_ps(&matrix[i][j]);
        // Multiply: multiplier * pivot_row
        __m256 vx = _mm256_mul_ps(vaik, vakj);
        // Subtract: target_row - (multiplier * pivot_row)
        vaij = _mm256_sub_ps(vaij, vx);
        // Store back 8 elements (using unaligned store for safety)
        _mm256_storeu_ps(&matrix[i][j], vaij);
    }
    
    // Handle remaining elements
    for (; j < n; j++) {
        matrix[i][j] = matrix[i][j] - factor * matrix[k][j];
    }
    
    matrix[i][k] = 0;
}

// Serial Gaussian elimination algorithm
void gaussEliminationSerial() {
    cout << "Starting serial Gaussian elimination..." << endl;
    
    for (int k = 0; k < n; k++) {
        // Normalize current row
        division_avx(k);
        
        // Eliminate subsequent rows
        for (int i = k + 1; i < n; i++) {
            elimination_avx(k, i);
        }
    }
    
    cout << "Serial Gaussian elimination completed" << endl;
}

// 1. Dynamic thread version elimination thread function
void* dynamicThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int i = k + t_id + 1;
    
    // Check bounds for safety
    if (i < n) {
        // Eliminate corresponding row
        elimination_avx(k, i);
    }
    
    pthread_exit(NULL);
}

// 1. Dynamic thread version Gaussian elimination
void gaussEliminationDynamicThread() {
    cout << "Starting dynamic thread Gaussian elimination..." << endl;
    
    for (int k = 0; k < n; k++) {
        // Normalize current row
        division_avx(k);
        
        // Create required number of threads
        int worker_count = n - 1 - k;
        if (worker_count <= 0) continue;
        
        pthread_t* handles = new pthread_t[worker_count];
        threadParam_t* param = new threadParam_t[worker_count];
        
        // Assign tasks to each thread
        for (int t_id = 0; t_id < worker_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        
        // Create threads
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, dynamicThreadFunc, &param[t_id]);
        }
        
        // Wait for all threads to complete
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        
        // Free resources
        delete[] handles;
        delete[] param;
    }
    
    cout << "Dynamic thread Gaussian elimination completed" << endl;
}

// 2. Static thread + semaphore synchronization thread function
void* staticSemaphoreThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]); // Wait for main thread notification
        
        // Perform AVX optimized elimination
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            elimination_avx(k, i);
        }
        
        sem_post(&sem_main); // Notify main thread completion
        sem_wait(&sem_workerend[t_id]); // Wait for all worker threads to complete current round
    }
    
    pthread_exit(NULL);
}

// 2. Static thread + semaphore synchronization Gaussian elimination
void gaussEliminationStaticSemaphore() {
    cout << "Starting static semaphore Gaussian elimination with " << NUM_THREADS << " threads..." << endl;
    
    // Initialize semaphores using the dynamic arrays
    sem_init(&sem_main, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    
    // Create threads
    pthread_t* handles = new pthread_t[NUM_THREADS];
    threadParam_t* param = new threadParam_t[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, staticSemaphoreThreadFunc, &param[t_id]);
    }
    
    // Main thread controls computation process
    for (int k = 0; k < n; k++) {
        // Normalize current row
        division_avx(k);
        
        // Wake up worker threads
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workerstart[t_id]);
        }
        
        // Wait for all worker threads to complete elimination
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        
        // Notify worker threads to enter next round
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workerend[t_id]);
        }
    }
    
    // Wait for all threads to finish
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    
    // Cleanup
    delete[] handles;
    delete[] param;
    
    cout << "Static semaphore Gaussian elimination completed" << endl;
}

// 3. Static thread + semaphore sync + three-loop thread function
void* staticFullThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // Thread 0 is responsible for normalization
        if (t_id == 0) {
            division_avx(k);
        } else {
            sem_wait(&sem_division[t_id-1]); // Non-0 threads wait for normalization to complete
        }
        
        // Thread 0 notifies other threads that normalization is complete
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_division[i]);
            }
        }
        
        // All threads perform elimination operations
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            elimination_avx(k, i);
        }
        
        // Thread synchronization, ensure all elimination operations are complete
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_wait(&sem_leader); // Wait for other threads to complete elimination
            }
            
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_elimination[i]); // Notify other threads to enter next round
            }
        } else {
            sem_post(&sem_leader); // Notify main thread completion
            sem_wait(&sem_elimination[t_id-1]); // Wait to enter next round
        }
    }
    
    pthread_exit(NULL);
}

// 3. Static thread + semaphore sync + three-loop Gaussian elimination
void gaussEliminationStaticFull() {
    cout << "Starting static full thread Gaussian elimination..." << endl;
    
    // Initialize semaphores
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        sem_init(&sem_division[i], 0, 0);
        sem_init(&sem_elimination[i], 0, 0);
    }
    
    // Create threads
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, staticFullThreadFunc, &param[t_id]);
    }
    
    // Wait for all threads to finish
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    
    // Destroy semaphores
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        sem_destroy(&sem_division[i]);
        sem_destroy(&sem_elimination[i]);
    }
    
    cout << "Static full thread Gaussian elimination completed" << endl;
}

// 4. Static thread + barrier synchronization thread function
void* staticBarrierThreadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    
    for (int k = 0; k < n; k++) {
        // Thread 0 is responsible for normalization
        if (t_id == 0) {
            division_avx(k);
        }
        
        // Use barrier synchronization to ensure normalization is complete
        pthread_barrier_wait(&barrier_division);
        
        // All threads perform elimination operations
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            elimination_avx(k, i);
        }
        
        // Use barrier synchronization to ensure all elimination operations are complete
        pthread_barrier_wait(&barrier_elimination);
    }
    
    pthread_exit(NULL);
}

// 4. Static thread + barrier synchronization Gaussian elimination
void gaussEliminationBarrier() {
    cout << "Starting barrier synchronization Gaussian elimination..." << endl;
    
    // Initialize barriers
    pthread_barrier_init(&barrier_division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_elimination, NULL, NUM_THREADS);
    
    // Create threads
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, staticBarrierThreadFunc, &param[t_id]);
    }
    
    // Wait for all threads to finish
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    
    // Destroy barriers
    pthread_barrier_destroy(&barrier_division);
    pthread_barrier_destroy(&barrier_elimination);
    
    cout << "Barrier Gaussian elimination completed" << endl;
}

// Check if the result is correct (compare with serial algorithm)
bool check_result(float** result_matrix) {
    cout << "Checking result correctness..." << endl;
    
    // Create a copy for serial computation
    float** temp_matrix = new float*[n];
    for (int i = 0; i < n; i++) {
        temp_matrix[i] = new float[n];
        memcpy(temp_matrix[i], matrix[i], n * sizeof(float));
    }
    
    // Run standard serial algorithm
    for (int k = 0; k < n; k++) {
        // Normalization
        float pivot = temp_matrix[k][k];
        for (int j = k + 1; j < n; j++) {
            temp_matrix[k][j] /= pivot;
        }
        temp_matrix[k][k] = 1.0;
        
        // Elimination
        for (int i = k + 1; i < n; i++) {
            float factor = temp_matrix[i][k];
            for (int j = k + 1; j < n; j++) {
                temp_matrix[i][j] -= factor * temp_matrix[k][j];
            }
            temp_matrix[i][k] = 0;
        }
    }
    
    // Compare results
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
    
    // Free temporary matrix
    for (int i = 0; i < n; i++) {
        delete[] temp_matrix[i];
    }
    delete[] temp_matrix;
    
    cout << "Result check completed: " << (correct ? "CORRECT" : "INCORRECT") << endl;
    return correct;
}

// Save result
float** save_result() {
    float** result = new float*[n];
    for (int i = 0; i < n; i++) {
        result[i] = new float[n];
        memcpy(result[i], matrix[i], n * sizeof(float));
    }
    return result;
}

// Restore original matrix
void restore_matrix(float** result) {
    for (int i = 0; i < n; i++) {
        memcpy(matrix[i], result[i], n * sizeof(float));
    }
}

// Free result matrix
void free_result(float** result) {
    for (int i = 0; i < n; i++) {
        delete[] result[i];
    }
    delete[] result;
}

// Calculate execution time
long long get_execution_time() {
    return (end_time.tv_sec - start_time.tv_sec) * 1000000LL + (end_time.tv_usec - start_time.tv_usec);
}

// Check if AVX is available
bool check_avx_support() {
    #if defined(__AVX__)
        return true;
    #else
        return false;
    #endif
}

int main(int argc, char** argv) {
    cout << "Starting Gaussian Elimination program with AVX optimizations" << endl;
    
    // Check AVX support
    if (!check_avx_support()) {
        cout << "AVX is not supported on this CPU. This program requires AVX instructions." << endl;
        cout << "Compile with -mavx flag and run on a CPU with AVX support." << endl;
        return -1;
    } else {
        cout << "AVX support confirmed" << endl;
    }

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <matrix_size> [num_threads]" << endl;
        return -1;
    }
    
    // Set matrix size
    n = atoi(argv[1]);
    if (n <= 0) {
        cout << "Invalid matrix size" << endl;
        return -1;
    }
    
    // Set thread count if provided
    if (argc >= 3) {
        NUM_THREADS = atoi(argv[2]);
        if (NUM_THREADS <= 0) {
            cout << "Invalid thread count. Using default (4)" << endl;
            NUM_THREADS = 4;
        }
    }
    
    cout << "Running Gaussian Elimination with AVX on " << n << "x" << n << " matrix" << endl;
    cout << "Number of threads: " << NUM_THREADS << endl;
    
    // Initialize random number generator
    srand(42);
    
    // Initialize matrix and synchronization primitives
    try {
        init_matrix();
        init_sync_primitives();
    } catch (const std::exception& e) {
        cout << "Error during initialization: " << e.what() << endl;
        return -1;
    }
    
    // Save original matrix
    float** original_matrix = nullptr;
    try {
        original_matrix = save_result();
    } catch (const std::exception& e) {
        cout << "Error saving original matrix: " << e.what() << endl;
        free_matrix();
        return -1;
    }
    
    // Run and test serial version
    try {
        cout << "Starting serial version..." << endl;
        gettimeofday(&start_time, NULL);
        gaussEliminationSerial();
        gettimeofday(&end_time, NULL);
        long long serial_time = get_execution_time();
        
        cout << "Serial version execution time: " << serial_time << " us" << endl;
        
        float** serial_result = save_result();
        
        // Run and test version 1: Dynamic thread version
        cout << "Starting dynamic thread version..." << endl;
        restore_matrix(original_matrix);
        gettimeofday(&start_time, NULL);
        gaussEliminationDynamicThread();
        gettimeofday(&end_time, NULL);
        long long dynamic_thread_time = get_execution_time();
        
        cout << "Dynamic Thread version execution time: " << dynamic_thread_time << " us" << endl;
        cout << "Dynamic Thread version speedup: " << (float)serial_time / dynamic_thread_time << endl;
        cout << "Dynamic Thread version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
        
        // Run and test version 2: Static thread + semaphore synchronization
        cout << "Starting static semaphore version..." << endl;
        restore_matrix(original_matrix);
        gettimeofday(&start_time, NULL);
        gaussEliminationStaticSemaphore();
        gettimeofday(&end_time, NULL);
        long long static_semaphore_time = get_execution_time();
        
        cout << "Static Semaphore version execution time: " << static_semaphore_time << " us" << endl;
        cout << "Static Semaphore version speedup: " << (float)serial_time / static_semaphore_time << endl;
        cout << "Static Semaphore version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
        
        // Run and test version 3: Static thread + semaphore sync + three-loop
        cout << "Starting static full thread version..." << endl;
        restore_matrix(original_matrix);
        gettimeofday(&start_time, NULL);
        gaussEliminationStaticFull();
        gettimeofday(&end_time, NULL);
        long long static_full_time = get_execution_time();
        
        cout << "Static Full Thread version execution time: " << static_full_time << " us" << endl;
        cout << "Static Full Thread version speedup: " << (float)serial_time / static_full_time << endl;
        cout << "Static Full Thread version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
        
        // Run and test version 4: Static thread + barrier synchronization
        cout << "Starting barrier version..." << endl;
        restore_matrix(original_matrix);
        gettimeofday(&start_time, NULL);
        gaussEliminationBarrier();
        gettimeofday(&end_time, NULL);
        long long barrier_time = get_execution_time();
        
        cout << "Barrier version execution time: " << barrier_time << " us" << endl;
        cout << "Barrier version speedup: " << (float)serial_time / barrier_time << endl;
        cout << "Barrier version correct: " << (check_result(serial_result) ? "YES" : "NO") << endl;
        
        // Output CSV format execution time
        cout << "\nCSV Format for plotting:\n";
        cout << "matrix_size,thread_count,serial,dynamic_thread,static_semaphore,static_full,barrier\n";
        cout << n << "," << NUM_THREADS << "," << serial_time << "," << dynamic_thread_time << "," 
             << static_semaphore_time << "," << static_full_time << "," << barrier_time << endl;
        
        // Output CSV format speedup
        cout << "\nSpeedup CSV Format for plotting:\n";
        cout << "matrix_size,thread_count,dynamic_thread,static_semaphore,static_full,barrier\n";
        cout << n << "," << NUM_THREADS << "," << (float)serial_time / dynamic_thread_time << "," 
             << (float)serial_time / static_semaphore_time << "," 
             << (float)serial_time / static_full_time << "," 
             << (float)serial_time / barrier_time << endl;
        
        // Free memory
        free_result(original_matrix);
        free_result(serial_result);
        
    } catch (const std::exception& e) {
        cout << "Error during computation: " << e.what() << endl;
        if (original_matrix != nullptr) {
            free_result(original_matrix);
        }
        free_matrix();
        return -1;
    }
    
    // Free matrix
    free_matrix();
    
    cout << "Program completed successfully" << endl;
    return 0;
}