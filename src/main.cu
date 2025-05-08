



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

const int HDR_DEPTH = 256;
const int d_utxo_set_idx4host_BYTES = 4;
const int STAKE_MODIFIER_BYTES = 32;
const int WALLET_UTXOS_HASH_BYTES = 32; // Will be more than 1 million    
const int WALLET_UTXOS_N_BYTES = 4; // Will be more than 1 million   
const int WALLET_UTXOS_TIME_FROM_BYTES = 4; // Will be more than 1 million  
const int START_TIME_BYTES = 4;
const int HASH_MERKLE_ROOT_BYTES = 32; 
const int HASH_PREV_BLOCK_BYTES = 32; 
const int N_BITS_BYTES = 4; 
const int N_TIME_BYTES = 4; 
const int PREV_STAKE_HASH_BYTES = 32; 
const int PREV_STAKE_N_BYTES = 4; 
const int BLOCK_SIG_BYTES = 80; // 1st byte is length   either 78 or 79, then normal vchsig(starts with 0x30 then total_len-2   then 8 nonce bytes)  use 80 for nice round number


// LARGE array holding the wallet info of hash and n
const int WALLET_UTXOS_LENGTH = 2000000; // Will be more than 1 million



/**************************** DATA TYPES ****************************/


typedef struct {
    uint64_t align1;
    uint8_t h_utxos_hash[WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH];
    uint64_t align2;
    uint8_t h_utxos_n[WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH];
    uint64_t align3;
    uint8_t h_utxos_block_from_time[WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH];
    uint64_t align12;
    uint8_t h_start_time[START_TIME_BYTES];
    uint64_t align11;
    uint8_t h_stake_modifier[STAKE_MODIFIER_BYTES];    
    uint64_t align4;
    uint8_t h_hash_merkle_root[HASH_MERKLE_ROOT_BYTES*HDR_DEPTH];
    uint64_t align5;
    volatile uint8_t h_hash_prev_block[HASH_PREV_BLOCK_BYTES*HDR_DEPTH];
    uint64_t align6;
    uint8_t h_n_bits[N_BITS_BYTES*HDR_DEPTH];
    uint64_t align7;
    uint8_t h_n_time[N_TIME_BYTES*HDR_DEPTH];
    uint64_t align8;
    uint8_t h_prev_stake_hash[PREV_STAKE_HASH_BYTES*HDR_DEPTH];
    uint64_t align9;
    uint8_t h_prev_stake_n[PREV_STAKE_N_BYTES*HDR_DEPTH];
    uint64_t align10;
    uint8_t h_block_sig[BLOCK_SIG_BYTES*HDR_DEPTH];
} STAGE1_S;








typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} CUDA_SHA256_CTX;


struct SharedData {
    volatile bool is_stage1;
    volatile uint64_t nonce;
    volatile uint8_t data[TOTAL_BYTES_SEND];      // Buffer to send data
    volatile uint32_t utxo_set_idx4host;
    volatile uint32_t utxo_set_time4host;
    bool is_data_ready;  // Flag to indicate if data is ready
    STAGE1_S stage1_data;
};



WORD nonce[1] = {0};


extern __global__ void cuda_miner(BYTE* d_gpu_num, BYTE* d_is_stage1, BYTE* key_data, BYTE* ctx_data, BYTE* hash_no_sig_in, BYTE* nonce4host, BYTE* d_utxo_set_idx4host, BYTE* d_utxo_set_time4host, BYTE* d_stake_modifier, BYTE* d_utxos_block_from_time, BYTE* d_utxos_hash, BYTE* d_utxos_n, BYTE* d_start_time,
                    BYTE* d_hash_merkle_root, BYTE* d_hash_prev_block, BYTE* d_n_bits, BYTE* d_n_time, BYTE* d_prev_stake_hash, BYTE* d_prev_stake_n, BYTE* d_block_sig );

// Define a struct to represent a uint256 (256-bit integer)
struct uint256 {
    uint64_t data[4];  // Array to hold four 64-bit parts
};

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

        //////////////////////STAGE1==================

    uint8_t *d_utxo_set_idx4host;
    uint8_t *h_utxo_set_idx4host = new uint8_t[d_utxo_set_idx4host_BYTES];   
    uint8_t *d_utxo_set_time4host;
    uint8_t *h_utxo_set_time4host = new uint8_t[d_utxo_set_idx4host_BYTES];       

    uint8_t *d_utxos_block_from_time;
    uint8_t *h_utxos_block_from_time = new uint8_t[WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH];   


    uint8_t *d_utxos_hash;
    uint8_t *h_utxos_hash = new uint8_t[WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH]; // 256 bits per hash

 
    uint8_t *d_utxos_n;
    uint8_t *h_utxos_n = new uint8_t[WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH]; 


    uint8_t *d_start_time;
    uint8_t *h_start_time = new uint8_t[START_TIME_BYTES];



    uint8_t *d_hash_merkle_root;
    uint8_t *h_hash_merkle_root = new uint8_t[HASH_MERKLE_ROOT_BYTES*HDR_DEPTH]; // 256 bits per hash


    uint8_t *d_hash_prev_block;
    uint8_t *h_hash_prev_block = new uint8_t[HASH_PREV_BLOCK_BYTES*HDR_DEPTH]; // 256 bits per hash


    uint8_t *d_n_bits;
    uint8_t *h_n_bits = new uint8_t[N_BITS_BYTES*HDR_DEPTH]; 


    uint8_t *d_n_time;
    uint8_t *h_n_time = new uint8_t[N_TIME_BYTES*HDR_DEPTH]; 


    uint8_t *d_prev_stake_hash;
    uint8_t *h_prev_stake_hash = new uint8_t[PREV_STAKE_HASH_BYTES*HDR_DEPTH];


    uint8_t *d_prev_stake_n;
    uint8_t *h_prev_stake_n = new uint8_t[PREV_STAKE_N_BYTES*HDR_DEPTH];  


    uint8_t *d_block_sig;
    uint8_t *h_block_sig = new uint8_t[BLOCK_SIG_BYTES*HDR_DEPTH]; 

    
    uint8_t *d_stake_modifier;
    uint8_t *h_stake_modifier = new uint8_t[STAKE_MODIFIER_BYTES]; 
    ///////////////////////////////////////////////////////////////////////////


    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, deviceId);

    std::cout << "BTCW GPU MINER RELEASE v26.4.71 - specify GPU number - 133 utxos per loop fast scan Dec 25 2024" << std::endl;

    std::cout << "Max threads per block: " << deviceProps.maxThreadsPerBlock << std::endl;


    // For a 1D grid:
    printf("Max grid size in X: %d\n", deviceProps.maxGridSize[0]); // x-dimension
    printf("Max grid size in Y: %d\n", deviceProps.maxGridSize[1]); // y-dimension
    printf("Max grid size in Z: %d\n", deviceProps.maxGridSize[2]); // z-dimension

    // Allocate memory on the device
    cudaMalloc(&d_gpu_num, 1);


    // Allocate memory on the device
    cudaMalloc(&d_is_stage1, 1);
    memset(h_is_stage1, 1, 1); // initizlize in stage1 on gpu side

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

    // Initialize data on the host (CPU)
    for (int i = 0; i < NONCE_SIZE_BYTES; ++i) {
        h_nonce_data[i] = 0;
    }    


    // Allocate memory on the device
    cudaMalloc(&d_utxo_set_idx4host, d_utxo_set_idx4host_BYTES);
    cudaMalloc(&d_utxo_set_time4host, d_utxo_set_idx4host_BYTES);
    // Initialize data on the host (CPU)
    for (int i = 0; i < d_utxo_set_idx4host_BYTES; ++i) {
        h_utxo_set_idx4host[i] = 0;
        h_utxo_set_time4host[i] = 0;
    }   


    // Initialize data on the host (CPU)
    for (int i = 0; i < START_TIME_BYTES; ++i) {
        h_start_time[i] = 0;
    }   

    // Initialize data on the host (CPU)
    for (int i = 0; i < STAKE_MODIFIER_BYTES; ++i) {
        h_stake_modifier[i] = 5;
    }       

    


    

    // Allocate memory on the device
    cudaMalloc(&d_stake_modifier, STAKE_MODIFIER_BYTES);
    cudaMalloc(&d_utxos_block_from_time, WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH);
    cudaMalloc(&d_utxos_hash, WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH);
    cudaMalloc(&d_utxos_n, WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH);
    cudaMalloc(&d_start_time, START_TIME_BYTES);
    cudaMalloc(&d_hash_merkle_root, HASH_MERKLE_ROOT_BYTES*HDR_DEPTH);
    cudaMalloc(&d_hash_prev_block, HASH_PREV_BLOCK_BYTES*HDR_DEPTH);
    cudaMalloc(&d_n_bits, N_BITS_BYTES*HDR_DEPTH);
    cudaMalloc(&d_n_time, N_TIME_BYTES*HDR_DEPTH);
    cudaMalloc(&d_prev_stake_hash, PREV_STAKE_HASH_BYTES*HDR_DEPTH);
    cudaMalloc(&d_prev_stake_n, PREV_STAKE_N_BYTES*HDR_DEPTH);
    cudaMalloc(&d_block_sig, BLOCK_SIG_BYTES*HDR_DEPTH); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 
    cudaMalloc(&d_stake_modifier, STAKE_MODIFIER_BYTES);



    // Copy the data to the GPU
    cudaMemcpy(d_gpu_num, h_gpu_num, 1, cudaMemcpyHostToDevice);

    // Stage2 stuff  
    cudaMemcpy(d_is_stage1, h_is_stage1, 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce_data, h_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxo_set_idx4host, h_utxo_set_idx4host, d_utxo_set_idx4host_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxo_set_time4host, h_utxo_set_time4host, d_utxo_set_idx4host_BYTES, cudaMemcpyHostToDevice);

    // Stage1 stuff
    
    cudaMemcpy(d_utxos_block_from_time, h_utxos_block_from_time, WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxos_hash, h_utxos_hash, WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxos_n, h_utxos_n, WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_time, h_start_time, START_TIME_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_merkle_root, h_hash_merkle_root, HASH_MERKLE_ROOT_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_prev_block, h_hash_prev_block, HASH_PREV_BLOCK_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_bits, h_n_bits, N_BITS_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_time, h_n_time, N_TIME_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_stake_hash, h_prev_stake_hash, PREV_STAKE_HASH_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_stake_n, h_prev_stake_n, PREV_STAKE_N_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_sig, h_block_sig, BLOCK_SIG_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 
    cudaMemcpy(d_stake_modifier, h_stake_modifier, STAKE_MODIFIER_BYTES, cudaMemcpyHostToDevice); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 



    // Create a stream for asynchronous operations
    cudaStream_t stream, kernel_stream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&kernel_stream);

    // Tell the miner which GPU number it is
    cudaMemcpyAsync(d_gpu_num, h_gpu_num, 1, cudaMemcpyHostToDevice, stream);

    // Copy the modified data from the host back to the GPU asynchronously  
    cudaMemcpyAsync(d_is_stage1, h_is_stage1, 1, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nonce_data, h_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_utxo_set_idx4host, h_utxo_set_idx4host, d_utxo_set_idx4host_BYTES, cudaMemcpyHostToDevice, stream);
    

    // Stage1 stuff

    cudaMemcpyAsync(d_utxos_block_from_time, h_utxos_block_from_time, WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_utxos_hash, h_utxos_hash, WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_utxos_n, h_utxos_n, WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_start_time, h_start_time, START_TIME_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_hash_merkle_root, h_hash_merkle_root, HASH_MERKLE_ROOT_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_hash_prev_block, h_hash_prev_block, HASH_PREV_BLOCK_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_n_bits, h_n_bits, N_BITS_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_n_time, h_n_time, N_TIME_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_prev_stake_hash, h_prev_stake_hash, PREV_STAKE_HASH_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_prev_stake_n, h_prev_stake_n, PREV_STAKE_N_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_sig, h_block_sig, BLOCK_SIG_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(d_stake_modifier, h_stake_modifier, STAKE_MODIFIER_BYTES, cudaMemcpyHostToDevice, stream); 


    // Wait for the kernel to complete
    cudaStreamSynchronize(stream);



    cudaMemcpy(d_utxos_block_from_time, h_utxos_block_from_time, WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxos_hash, h_utxos_hash, WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxos_n, h_utxos_n, WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_time, h_start_time, START_TIME_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_merkle_root, h_hash_merkle_root, HASH_MERKLE_ROOT_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_prev_block, h_hash_prev_block, HASH_PREV_BLOCK_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_bits, h_n_bits, N_BITS_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_time, h_n_time, N_TIME_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_stake_hash, h_prev_stake_hash, PREV_STAKE_HASH_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_stake_n, h_prev_stake_n, PREV_STAKE_N_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_sig, h_block_sig, BLOCK_SIG_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 
    cudaMemcpy(d_stake_modifier, h_stake_modifier, STAKE_MODIFIER_BYTES, cudaMemcpyHostToDevice); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 
    //===========================================KERNEL======================================================
    // We are starting the KERNEL with NO DATA - This is intentional, data will be given to it on the fly from the BTCW node.
    cuda_miner<<<128, 256, 0, kernel_stream>>>(d_gpu_num, d_is_stage1, d_key_data, d_ctx_data, d_hash_no_sig_data, d_nonce_data, d_utxo_set_idx4host, d_utxo_set_time4host, d_stake_modifier, d_utxos_block_from_time, d_utxos_hash, d_utxos_n, d_start_time, d_hash_merkle_root, d_hash_prev_block, d_n_bits, d_n_time, d_prev_stake_hash, d_prev_stake_n, d_block_sig);

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
    uint256 hash_no_sig;
    unsigned char key_data[32];

    bool is_stage1 = true;


    uint64_t *p_data = (uint64_t *)shared_data->data;


    // Cast to the volatile pointer to ensure we don't optimize reads/writes
    volatile SharedData* mapped_data = (volatile SharedData*) shared_data;


    uint32_t throttle = 0x0;
    bool is_init = false;

    while ( true )
    {

        // one time init
        if ( !is_init )
        {
            cudaMemcpyAsync(d_utxos_block_from_time, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_utxos_block_from_time[0])), WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_utxos_hash, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_utxos_hash[0])), WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_utxos_n, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_utxos_n[0])), WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
            is_init = true;
        }
        
        if ( (throttle % 0x3) == 0 )
        {

        
            if ( shared_data->is_stage1 )
            {
                // we are in stage1
                printf("STAGE1 BLOCK DATA - CPU SIDE\n");

                
                cudaMemcpyAsync(d_start_time, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_start_time[0])), START_TIME_BYTES, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_stake_modifier, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_stake_modifier[0])), STAKE_MODIFIER_BYTES, cudaMemcpyHostToDevice, stream);
                
                cudaMemcpyAsync(d_hash_merkle_root, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_hash_merkle_root[0])), HASH_MERKLE_ROOT_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_hash_prev_block, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_hash_prev_block[0])), HASH_PREV_BLOCK_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_n_bits, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_n_bits[0])), N_BITS_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_n_time, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_n_time[0])), N_TIME_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_prev_stake_hash, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_prev_stake_hash[0])), PREV_STAKE_HASH_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_prev_stake_n, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_prev_stake_n[0])), PREV_STAKE_N_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_block_sig, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_block_sig[0])), BLOCK_SIG_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 


            }
            else
            {

                // we are in stage2

                //if ( is_stage1 )
                //{
                    // we need to update our state. this is a transition. Copy over the data for stage2.
                    is_stage1 = false;

                    //Host update the data, send it to the GPU
                    printf("STAGE2 BLOCK DATA - CPU SIDE\n");


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
                //}

            }

        }

        throttle++;

        // convert from bool to 1 byte
        if ( shared_data->is_stage1 )
        {
            h_is_stage1[0] = 0xFF;
        }
        else
        {
            h_is_stage1[0] = 0;
        }

        cudaMemcpyAsync(d_is_stage1, h_is_stage1, 1, cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->nonce)), d_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyDeviceToHost, stream); 

        cudaMemcpyAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->utxo_set_idx4host)), d_utxo_set_idx4host, d_utxo_set_idx4host_BYTES, cudaMemcpyDeviceToHost, stream); 
        cudaMemcpyAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->utxo_set_time4host)), d_utxo_set_time4host, d_utxo_set_idx4host_BYTES, cudaMemcpyDeviceToHost, stream); 


        usleep(500000);

    }


    // Wait for the kernel to complete
    cudaStreamSynchronize(kernel_stream);



    // Cleanup


    cudaFree(d_gpu_num);
    delete[] h_gpu_num;

    cudaFree(d_is_stage1);
    delete[] h_is_stage1;

    cudaFree(d_ctx_data);
    delete[] h_ctx_data;

    cudaFree(d_key_data);
    delete[] h_key_data;

    cudaFree(d_hash_no_sig_data);
    delete[] h_hash_no_sig_data;

    cudaFree(d_nonce_data);
    delete[] h_nonce_data;

    cudaFree(d_utxo_set_idx4host);
    delete[] h_utxo_set_idx4host;

    cudaFree(d_stake_modifier);
    delete[] h_stake_modifier;

    cudaFree(d_utxos_hash);
    delete[] h_utxos_hash;

    cudaFree(d_utxos_block_from_time);
    delete[] h_utxos_block_from_time;


    cudaFree(d_utxos_n);
    delete[] h_utxos_n;

    cudaFree(d_start_time);
    delete[] h_start_time;


    cudaFree(d_hash_merkle_root);
    delete[] h_hash_merkle_root;

    cudaFree(d_hash_prev_block);
    delete[] h_hash_prev_block;

    cudaFree(d_n_bits);
    delete[] h_n_bits;

    cudaFree(d_n_time);
    delete[] h_n_time;

    cudaFree(d_prev_stake_hash);
    delete[] h_prev_stake_hash;

    cudaFree(d_prev_stake_n);
    delete[] h_prev_stake_n;

    cudaFree(d_block_sig);
    delete[] h_block_sig;



    cudaStreamDestroy(stream);
    cudaStreamDestroy(kernel_stream);


    munmap(shared_data, sizeof(SharedData));
    close(shm_fd);    

    return 0;
}

