// shm_mutex.c - 编译为共享库
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

#define MAGIC_NUMBER 0x4D554F4E  // "MUON"
#define VERSION 1
#define CACHE_LINE_SIZE 64

typedef struct {
    int magic;
    int version;
    int num_buffers;
    char padding[CACHE_LINE_SIZE - 12]; // 填充到缓存行
} shm_header_t;

typedef struct {
    int state;
    pthread_mutex_t mutex;
    char padding[CACHE_LINE_SIZE - sizeof(int) - sizeof(pthread_mutex_t)];
} buffer_control_t;

// 初始化共享内存和mutex
int init_shared_memory(const char* control_shm_name, const char* data_shm_name, 
                      int num_buffers, size_t data_size) {
    // 创建控制块共享内存
    int control_fd = shm_open(control_shm_name, O_CREAT | O_RDWR, 0666);
    if (control_fd == -1) return -1;
    
    size_t control_size = sizeof(shm_header_t) + num_buffers * sizeof(buffer_control_t);
    ftruncate(control_fd, control_size);
    
    void* control_addr = mmap(NULL, control_size, PROT_READ | PROT_WRITE, 
                             MAP_SHARED, control_fd, 0);
    if (control_addr == MAP_FAILED) return -1;
    
    // 初始化头部
    shm_header_t* header = (shm_header_t*)control_addr;
    header->magic = MAGIC_NUMBER;
    header->version = VERSION;
    header->num_buffers = num_buffers;
    
    // 初始化mutex
    buffer_control_t* controls = (buffer_control_t*)(header + 1);
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
    
    for (int i = 0; i < num_buffers; i++) {
        controls[i].state = 0; // 初始状态为空
        pthread_mutex_init(&controls[i].mutex, &attr);
    }
    
    pthread_mutexattr_destroy(&attr);
    
    // 创建数据块共享内存
    int data_fd = shm_open(data_shm_name, O_CREAT | O_RDWR, 0666);
    if (data_fd == -1) return -1;
    
    ftruncate(data_fd, data_size);
    
    close(control_fd);
    close(data_fd);
    munmap(control_addr, control_size);
    
    return 0;
}

// 连接到共享内存
void* attach_shared_memory(const char* shm_name, size_t size) {
    int fd = shm_open(shm_name, O_RDWR, 0666);
    if (fd == -1) return NULL;
    
    void* addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    
    return addr == MAP_FAILED ? NULL : addr;
}

// mutex操作包装
int lock_mutex(pthread_mutex_t* mutex) {
    return pthread_mutex_lock(mutex);
}

int try_lock_mutex(pthread_mutex_t* mutex) {
    return pthread_mutex_trylock(mutex);
}

int unlock_mutex(pthread_mutex_t* mutex) {
    return pthread_mutex_unlock(mutex);
}

int recover_mutex(pthread_mutex_t* mutex) {
    return pthread_mutex_consistent(mutex);
}