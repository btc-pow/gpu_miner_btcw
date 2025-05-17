



#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>

#include <iostream>
#include <vector>
#include <random>
#include <cuda.h>
#include <cstring>  // For memcpy

#include "secp256k1.h"

#include <ncurses.h>

#define SHM_NAME "/shared_mem"


const int CTX_SIZE_BYTES = 8*20; // 160
const int KEY_SIZE_BYTES = 32;
const int HASH_NO_SIG_SIZE_BYTES = 32;
const int TOTAL_BYTES_SEND = CTX_SIZE_BYTES + KEY_SIZE_BYTES + HASH_NO_SIG_SIZE_BYTES;



/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

/**************************** DATA TYPES ****************************/
typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} CUDA_SHA256_CTX;


struct SharedData {
    volatile uint64_t nonce;
    volatile uint8_t data[TOTAL_BYTES_SEND];      // Buffer to send data
};



//extern __global__ void cuda_miner(BYTE* d_gpu_num, BYTE* key_data, BYTE* ctx_data, BYTE* hash_no_sig_in, BYTE* nonce4host, BYTE* nonce4hashrate );

// Define a struct to represent a uint256 (256-bit integer)
struct uint256 {
    uint64_t data[4];  // Array to hold four 64-bit parts
};


#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>



volatile uint64_t nonce4hashrate = 0;
volatile uint64_t nonce4hashrate_prev = 0;

int main( int argc, char* argv[] ) {

    int gpu_num = 0; // default
    if ( argc == 2 )
    {
        gpu_num = (int)atoi(argv[1]);
    }


    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;

    // Initialize the CUDA driver API
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Load the PTX module
    CUresult res = cuModuleLoad(&cuModule, "kernel.ptx");
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to load PTX\n";
        return 1;
    }

    // Get the kernel function
    // Use the mangled name
    // cat kernel.ptx  | grep cuda_miner
	// .globl	_Z10cuda_minerPhS_S_S_S_
    cuModuleGetFunction(&cuFunction, cuModule, "_Z10cuda_minerPhS_S_S_S_S_");


    const int CTX_SIZE_BYTES = 8*20; // 160
    const int KEY_SIZE_BYTES = 32;
    const int HASH_NO_SIG_SIZE_BYTES = 32;
    const int TOTAL_BYTES_SEND = CTX_SIZE_BYTES + KEY_SIZE_BYTES + HASH_NO_SIG_SIZE_BYTES;

    const int NONCE_SIZE_BYTES = 8;


    //uint8_t *d_gpu_num;
    // Allocate pinned host memory  
    void *h_gpu_num;
    cuMemHostAlloc(&h_gpu_num, 1, CU_MEMHOSTALLOC_PORTABLE);      
    *static_cast<uint8_t*>(h_gpu_num) = static_cast<uint8_t>(gpu_num);

    //////////////////////STAGE2==================

    //uint8_t *d_ctx_data;
    uint8_t *h_ctx_data = new uint8_t[CTX_SIZE_BYTES];


    //uint8_t *d_key_data;
    uint8_t *h_key_data = new uint8_t[KEY_SIZE_BYTES];    



    //uint8_t *d_hash_no_sig_data;
    uint8_t *h_hash_no_sig_data = new uint8_t[HASH_NO_SIG_SIZE_BYTES];      


    //uint8_t *d_nonce_data;
    uint8_t *h_nonce_data = new uint8_t[NONCE_SIZE_BYTES];      

    //uint8_t *d_nonce4hashrate_data;
    uint8_t *h_nonce4hashrate_data = new uint8_t[NONCE_SIZE_BYTES];   
    ///////////////////////////////////////////////////////////////////////////


    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, deviceId);

    std::cout << "BTCW GPU MINER RELEASE v26.5.99 - May 8 2025" << std::endl;

    std::cout << "Max threads per block: " << deviceProps.maxThreadsPerBlock << std::endl;


    // For a 1D grid:
    printf("Max grid size in X: %d\n", deviceProps.maxGridSize[0]); // x-dimension
    printf("Max grid size in Y: %d\n", deviceProps.maxGridSize[1]); // y-dimension
    printf("Max grid size in Z: %d\n", deviceProps.maxGridSize[2]); // z-dimension





    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Error getting CUDA device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA-capable device(s):\n";
    cudaDeviceProp prop;
    for (int i = 0; i < deviceCount; ++i) {
        //cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n"
                  << "  Compute capability: " << prop.major << "." << prop.minor << "\n"
                  << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n"
                  << "  Multiprocessors: " << prop.multiProcessorCount << "\n"
                  << "  Clock rate: " << prop.clockRate / 1000 << " MHz\n"
                  << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n"
                  << "  Memory Bus Width (bits): " << prop.memoryBusWidth << "\n"
                  << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n"
                  << "  Max grid size: [" << prop.maxGridSize[0] << ", "
                                         << prop.maxGridSize[1] << ", "
                                         << prop.maxGridSize[2] << "]\n"
                  << "  Max threads dim: [" << prop.maxThreadsDim[0] << ", "
                                          << prop.maxThreadsDim[1] << ", "
                                          << prop.maxThreadsDim[2] << "]\n\n";
    }



    // Allocate memory on the device
    //cudaMalloc(&d_gpu_num, 1);
    CUdeviceptr d_gpu_num;
    cuMemAlloc(&d_gpu_num, 1);    


    // Allocate memory on the device
    //cudaMalloc(&d_ctx_data, CTX_SIZE_BYTES);
    CUdeviceptr d_ctx_data;
    cuMemAlloc(&d_ctx_data, CTX_SIZE_BYTES);        


    // Allocate memory on the device
    //cudaMalloc(&d_key_data, KEY_SIZE_BYTES);
    CUdeviceptr d_key_data;
    cuMemAlloc(&d_key_data, KEY_SIZE_BYTES);       


    // Allocate memory on the device
    //cudaMalloc(&d_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES);
    CUdeviceptr d_hash_no_sig_data;
    cuMemAlloc(&d_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES);      
 

    // Allocate memory on the device
    //cudaMalloc(&d_nonce_data, NONCE_SIZE_BYTES);
    CUdeviceptr d_nonce_data;
    cuMemAlloc(&d_nonce_data, NONCE_SIZE_BYTES);   


    CUdeviceptr d_nonce4hashrate_data;
    cuMemAlloc(&d_nonce4hashrate_data, NONCE_SIZE_BYTES);   

    CUstream stream, kernel_stream;
    cuStreamCreate(&kernel_stream, 0);
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);


    //===========================================KERNEL======================================================
    // We are starting the KERNEL with NO DATA - This is intentional, data will be given to it on the fly from the BTCW node.
    //__global__ void cuda_miner<<<128, 256, 0, kernel_stream>>>(d_gpu_num, d_key_data, d_ctx_data, d_hash_no_sig_data, d_nonce_data, d_nonce4hashrate_data);


    void* args[] = {
        &d_gpu_num,
        &d_key_data,
        &d_ctx_data,
        &d_hash_no_sig_data,
        &d_nonce_data,
        &d_nonce4hashrate_data
    };    
    
    res = cuLaunchKernel(
        cuFunction,
        128, 1, 1,     // Grid dimensions
        256, 1, 1,     // Block dimensions
        0,             // Shared memory size
        kernel_stream, // Stream
        args,          // Kernel arguments
        nullptr        // Extra (usually null)
    );

    if (res != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(res, &errStr);
        std::cerr << "cuLaunchKernel failed: " << (errStr ? errStr : "Unknown error") << std::endl;
    }
    //=================================================================================================================


    // Open shared memory
    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Error opening shared memory" << std::endl;
        return 1;
    }

    // Map shared memory into the process's address space
    SharedData* shared_data = (SharedData*) mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_data == MAP_FAILED) {
        std::cerr << "Error mapping shared memory" << std::endl;
        return 1;
    }


    secp256k1_ecmult_gen_context ctx_obj;
    secp256k1_ecmult_gen_context *ctx = &ctx_obj;

    unsigned char key_data[32];


    uint64_t *p_data = (uint64_t *)shared_data->data;


    // Cast to the volatile pointer to ensure we don't optimize reads/writes
    volatile SharedData* mapped_data = (volatile SharedData*) shared_data;

    int hashrate = 10000;



    // Tell the miner which GPU number it is
    //cudaMemcpyAsync(d_gpu_num, h_gpu_num, 1, cudaMemcpyHostToDevice, stream);
    // Async copy device -> host
    cuMemcpyHtoDAsync(d_gpu_num, h_gpu_num, 1, stream);  

    uint32_t throttle = 0x0;
    initscr();            // Start ncurses mode
    noecho();             // Don't echo keypresses
    curs_set(FALSE);      // Hide the cursor

    int prev_y, prev_x;
    int curr_y, curr_x;
    getmaxyx(stdscr, prev_y, prev_x);  // Initial size
    mvprintw(0, 0, "Bitcoin-PoW GPU Miner v26.5.4\n");

    volatile uint64_t nonce_prev = 1234;
    nonce4hashrate_prev = 12345;
    static uint64_t hash_no_sig = 0;
    while ( true )
    {

    int changeCount = 0;

    const int durationSeconds = 2; // Run for 2 seconds
    auto startTime = std::chrono::steady_clock::now();        

        while (std::chrono::steady_clock::now() - startTime < std::chrono::seconds(durationSeconds)) 
        {        
            
            if ( (throttle % 0x3) == 0 )
            {

                //Host update the data, send it to the GPU
                //printf("STAGE2 BLOCK DATA - CPU SIDE\n");


                // Data set from BTCW node
                memcpy( &h_key_data[0], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[0])), 32 );


                memcpy( &h_ctx_data[0],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[32])), 8 );
                memcpy( &h_ctx_data[8],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[40])), 8 );
                memcpy( &h_ctx_data[16], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[48])), 8 );
                memcpy( &h_ctx_data[24], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[56])), 8 );

                memcpy( &h_ctx_data[32],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[64])), 8 );
                memcpy( &h_ctx_data[40],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[72])), 8 );
                memcpy( &h_ctx_data[48],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[80])), 8 );
                memcpy( &h_ctx_data[56],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[88])), 8 );
                memcpy( &h_ctx_data[64],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[96])), 8 );
                memcpy( &h_ctx_data[72],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[104])), 8 );
                memcpy( &h_ctx_data[80],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[112])), 8 );
                memcpy( &h_ctx_data[88],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[120])), 8 );
                memcpy( &h_ctx_data[96],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[128])), 8 );
                memcpy( &h_ctx_data[104], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[136])), 8 );
                memcpy( &h_ctx_data[112], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[144])), 8 );
                memcpy( &h_ctx_data[120], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[152])), 8 );
                memcpy( &h_ctx_data[128], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[160])), 8 );
                memcpy( &h_ctx_data[136], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[168])), 8 );
                memcpy( &h_ctx_data[144], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[176])), 8 );

                memcpy( &h_ctx_data[152], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[184])), 4 );
                memcpy( &h_ctx_data[156], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[188])), 4 );


                memcpy( &h_hash_no_sig_data[0],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), 8);
                memcpy( &h_hash_no_sig_data[8],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[200])), 8);
                memcpy( &h_hash_no_sig_data[16], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[208])), 8);
                memcpy( &h_hash_no_sig_data[24], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[216])), 8);


            // Copy the modified data from the host back to the GPU asynchronously  
            //cudaMemcpyAsync(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
            cuMemcpyHtoDAsync(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, stream);

            //cudaMemcpyAsync(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
            cuMemcpyHtoDAsync(d_key_data, h_key_data, KEY_SIZE_BYTES, stream);

            //cudaMemcpyAsync(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice, stream);   
            cuMemcpyHtoDAsync(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, stream); 

            }

            throttle++;

            cuMemcpyDtoHAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->nonce)), d_nonce_data, NONCE_SIZE_BYTES, stream);
            cuMemcpyDtoHAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&nonce4hashrate)), d_nonce4hashrate_data, NONCE_SIZE_BYTES, stream); 

            //printf("STAGE2 BLOCK DATA UPDATED - DEVICE\n");



            
            getmaxyx(stdscr, curr_y, curr_x);

            if (curr_y != prev_y || curr_x != prev_x) {
                clear(); // Screen size changed — clear and redraw
                prev_y = curr_y;
                prev_x = curr_x;
                mvprintw(0, 0, "Bitcoin-PoW GPU Miner v26.5.4\n");
            }

            if ( nonce_prev != shared_data->nonce )
            {
                nonce_prev = shared_data->nonce;
                mvprintw(2, 0, "Hash found - NONCE: %016llx\n", nonce_prev);
            }        


            
            memcpy( &hash_no_sig,  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), 8);
            // Always show the current hash no signature
            mvprintw(4, 0, "Hash no sig low64: %016llx\n", hash_no_sig);

            if ( hash_no_sig == 0 )
            {
                
                mvprintw(6, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
                mvprintw(7, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
                mvprintw(8, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");

                // Try to open shared memory again
                shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
                if (shm_fd == -1) {
                    //std::cerr << "Error opening shared memory" << std::endl;
                }
                else
                {
                    // Map shared memory into the process's address space
                    shared_data = (SharedData*) mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
                    if (shared_data == MAP_FAILED) {
                        //std::cerr << "Error mapping shared memory" << std::endl;
                    }
                }
                usleep(1000000);
                                
            }
            else
            {
                // remove warning
                mvprintw(6, 0, "CONNECTED TO BTCW NODE WALLET\n");
                mvprintw(7, 0, "\n");
                mvprintw(8, 0, "\n");
            }




            if (nonce4hashrate != nonce4hashrate_prev) {
                changeCount++;
                nonce4hashrate_prev = nonce4hashrate;
            }



            usleep(50);

        }


        double rate = static_cast<double>(changeCount) / durationSeconds;
        rate *= 65536;

        if ( hash_no_sig == 0 )
        {
            rate = 0;
        }        

        // Always show status at the bottom
        mvprintw(curr_y - 5, 0, "=======================================================\n");
        mvprintw(curr_y - 4, 0, "Device: %s\n", prop.name);
        mvprintw(curr_y - 3, 0, "-------------------------------------------------------\n");
        mvprintw(curr_y - 2, 0, "Hashrate: %lf H/s\n", rate);
        mvprintw(curr_y - 1, 0, "=======================================================\n");
        refresh();


    }




    // Cleanup


    cuMemFree(d_gpu_num);
    delete[] h_gpu_num;

    cuMemFree(d_ctx_data);
    delete[] h_ctx_data;

    cuMemFree(d_key_data);
    delete[] h_key_data;

    cuMemFree(d_hash_no_sig_data);
    delete[] h_hash_no_sig_data;

    cuMemFree(d_nonce_data);
    delete[] h_nonce_data;

    cuMemFree(d_nonce4hashrate_data);
    delete[] h_nonce4hashrate_data;

    cuStreamDestroy(stream);
    cuStreamDestroy(kernel_stream);


    munmap(shared_data, sizeof(SharedData));
    close(shm_fd);    

    return 0;
}

