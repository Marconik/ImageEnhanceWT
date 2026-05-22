cmake_minimum_required(VERSION 3.18)
project(WaveletCudaEnhance
    DESCRIPTION "GPU-accelerated wavelet-based image enhancement using Haar lifting scheme"
    LANGUAGES CXX CUDA
)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures (Turing:75, Ampere:80/86, Ada:89)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

# Find dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(CUDAToolkit REQUIRED)

# -------------------------------------------------------------------
# Executable: wavelet_enhance
# -------------------------------------------------------------------
add_executable(wavelet_enhance
    src/main.cpp
    src/wavelet_cuda.cu
)

target_include_directories(wavelet_enhance PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(wavelet_enhance PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
)

# MSVC: 强制使用 UTF-8 编码以支持中文注释和字符串
if(MSVC)
    # C++ 源文件: 直接添加 /utf-8
    target_compile_options(wavelet_enhance PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    # CUDA 源文件: 通过 -Xcompiler 将 /utf-8 传递给 host 编译器
    target_compile_options(wavelet_enhance PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# Optional: set separable compilation if device code gets large
# set_property(TARGET wavelet_enhance PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Print configuration summary
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA toolkit:   ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
