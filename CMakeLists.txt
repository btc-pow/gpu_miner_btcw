cmake_minimum_required(VERSION 3.16)

# Set the project name
project(gpu_miner)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



set(CMAKE_CUDA_ARCHITECTURES 89)  # For example, targeting NVIDIA Volta architecture (sm_89)


# Manually specify the path to CUDA toolkit if it's not in a standard location
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6/bin" CACHE PATH "Path to CUDA toolkit")

# Manually specify the path to nvcc (the CUDA compiler)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_80 --std=c++11 -g -G -O0")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 --std=c++11 -g -G -O0")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -use_fast_math -arch=sm_89 --std=c++11 -O3 ")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -arch=sm_89 --std=c++11 -O3 ")

# Enable CUDA support (Ensure CUDA is installed and the path is set)
#find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)


# Ensure the target uses both C++ and CUDA languages
enable_language(CXX CUDA)


# Specify where to find the headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Define the CUDA source files
set(SRC_FILES
    src/gpu_miner.cu
    src/main.cu
)


find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIR})





add_executable(gpu_miner ${SRC_FILES})

# Optional: Add any additional compile options
target_compile_options(gpu_miner PRIVATE -Wall -O3)
target_link_libraries(gpu_miner ${CURSES_LIBRARIES})


# PTX generation setup
set(PTX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel.ptx)
set(CU_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_miner.cu)


add_custom_command(
    OUTPUT ${PTX_OUTPUT}
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx -rdc=true
            --ptx ${CU_FILE} 
            -o ${PTX_OUTPUT}
    DEPENDS ${CU_FILE}
    COMMENT "Generating PTX file from ${CU_FILE}"
)

add_custom_target(generate_ptx ALL DEPENDS ${PTX_OUTPUT})

