



#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>

#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstring>  // For memcpy

#include "secp256k1.h"

#include <ncurses.h>

#define SHM_NAME "/shared_mem"
#define SEM_EMPTY "/sem_empty"
#define SEM_FULL "/sem_full"

//const int NUM_HASHES_BATCH = (1<<24); // This is really the total number of threads we want on the GPU (16M   )
const int NUM_HASHES_BATCH = (1); // This is really the total number of threads we want on the GPU (16M   )


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



WORD nonce[1] = {0};


extern __global__ void cuda_miner(BYTE* d_gpu_num, BYTE* key_data, BYTE* ctx_data, BYTE* hash_no_sig_in, BYTE* nonce4host, BYTE* nonce4hashrate );

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


    const int CTX_SIZE_BYTES = 8*20; // 160
    const int KEY_SIZE_BYTES = 32;
    const int HASH_NO_SIG_SIZE_BYTES = 32;
    const int TOTAL_BYTES_SEND = CTX_SIZE_BYTES + KEY_SIZE_BYTES + HASH_NO_SIG_SIZE_BYTES;

    const int NONCE_SIZE_BYTES = 8;


    uint8_t *d_gpu_num;
    uint8_t *h_gpu_num = new uint8_t[1];
    *h_gpu_num = gpu_num;

    uint8_t *d_is_stage1;
    uint8_t *h_is_stage1 = new uint8_t[1];
    //////////////////////STAGE2==================

    uint8_t *d_ctx_data;
    uint8_t *h_ctx_data = new uint8_t[CTX_SIZE_BYTES];


    uint8_t *d_key_data;
    uint8_t *h_key_data = new uint8_t[KEY_SIZE_BYTES];    



    uint8_t *d_hash_no_sig_data;
    uint8_t *h_hash_no_sig_data = new uint8_t[HASH_NO_SIG_SIZE_BYTES];      


    uint8_t *d_nonce_data;
    uint8_t *h_nonce_data = new uint8_t[NONCE_SIZE_BYTES];     

    uint8_t *d_nonce4hashrate_data;
    uint8_t *h_nonce4hashrate_data = new uint8_t[NONCE_SIZE_BYTES];   
    ///////////////////////////////////////////////////////////////////////////


    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, deviceId);


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
    cudaMalloc(&d_gpu_num, 1);


    // Allocate memory on the device
    cudaMalloc(&d_ctx_data, CTX_SIZE_BYTES);

    // Initialize data on the host (CPU)
    for (int i = 0; i < CTX_SIZE_BYTES; ++i) {
        h_ctx_data[i] = 0;
    }

    // Allocate memory on the device
    cudaMalloc(&d_key_data, KEY_SIZE_BYTES);

    // Initialize data on the host (CPU)
    for (int i = 0; i < KEY_SIZE_BYTES; ++i) {
        h_key_data[i] = 0;
    }

    // Allocate memory on the device
    cudaMalloc(&d_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES);

    // Initialize data on the host (CPU)
    for (int i = 0; i < HASH_NO_SIG_SIZE_BYTES; ++i) {
        h_hash_no_sig_data[i] = 0;
    }    


    // Allocate memory on the device
    cudaMalloc(&d_nonce_data, NONCE_SIZE_BYTES);
    cudaMalloc(&d_nonce4hashrate_data, NONCE_SIZE_BYTES);
    

    // Initialize data on the host (CPU)
    for (int i = 0; i < NONCE_SIZE_BYTES; ++i) {
        h_nonce_data[i] = 0;
        h_nonce4hashrate_data[i] = 0;
    }    

  
    // Copy the data to the GPU
    cudaMemcpy(d_gpu_num, h_gpu_num, 1, cudaMemcpyHostToDevice);

    // Stage2 stuff  
    cudaMemcpy(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce_data, h_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce4hashrate_data, h_nonce4hashrate_data, NONCE_SIZE_BYTES, cudaMemcpyHostToDevice);



    // Create a stream for asynchronous operations
    cudaStream_t stream, kernel_stream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&kernel_stream);

    // Tell the miner which GPU number it is
    cudaMemcpyAsync(d_gpu_num, h_gpu_num, 1, cudaMemcpyHostToDevice, stream);

    // Copy the modified data from the host back to the GPU asynchronously  
    cudaMemcpyAsync(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nonce_data, h_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nonce4hashrate_data, h_nonce4hashrate_data, NONCE_SIZE_BYTES, cudaMemcpyHostToDevice, stream);

    // Wait for the kernel to complete
    cudaStreamSynchronize(stream);


    //===========================================KERNEL======================================================
    // We are starting the KERNEL with NO DATA - This is intentional, data will be given to it on the fly from the BTCW node.
    cuda_miner<<<128, 256, 0, kernel_stream>>>(d_gpu_num, d_key_data, d_ctx_data, d_hash_no_sig_data, d_nonce_data, d_nonce4hashrate_data);
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
                cudaMemcpyAsync(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice, stream);    

            }

            throttle++;

            cudaMemcpyAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->nonce)), d_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyDeviceToHost, stream); 
            cudaMemcpyAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&nonce4hashrate)), d_nonce4hashrate_data, NONCE_SIZE_BYTES, cudaMemcpyDeviceToHost, stream); 

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
                mvprintw(6, 0, "\n");
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


    // Wait for the kernel to complete
    cudaStreamSynchronize(kernel_stream);



    // Cleanup


    cudaFree(d_gpu_num);
    delete[] h_gpu_num;

    cudaFree(d_ctx_data);
    delete[] h_ctx_data;

    cudaFree(d_key_data);
    delete[] h_key_data;

    cudaFree(d_hash_no_sig_data);
    delete[] h_hash_no_sig_data;

    cudaFree(d_nonce_data);
    delete[] h_nonce_data;


    cudaStreamDestroy(stream);
    cudaStreamDestroy(kernel_stream);


    munmap(shared_data, sizeof(SharedData));
    close(shm_fd);    

    return 0;
}

