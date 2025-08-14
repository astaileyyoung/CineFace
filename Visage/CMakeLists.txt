cmake_minimum_required(VERSION 3.10)
project(Visage CXX)

add_definitions(-D__STDC_CONSTANT_MACROS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc dnn cudacodec videoio cudaimgproc)
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

get_filename_component(_opencv_config "${OpenCV_DIR}/OpenCVConfig.cmake" ABSOLUTE)
message(STATUS "Using OpenCVConfig.cmake: ${_opencv_config}")

# Set CUDA toolkit path if not already set
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "Path to CUDA Toolkit")
endif()

# Explicitly set CUDA include and library paths for FindCUDA
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_CUDART_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")

# (Optional) Export these variables for CMake's find_package
set(ENV{CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR})
set(ENV{CUDA_PATH} ${CUDA_TOOLKIT_ROOT_DIR})

# CUDA runtime for libtorch
find_library(CUDA_CUDART_LIBRARY cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/local/cuda/lib64)

set(Torch_DIR /opt/libtorch/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(TENSOR_DIR "/opt/TensorRT-8.6.1.6")
link_directories(${TENSOR_DIR}/lib)
find_path(TensorRT_INCLUDE_DIR NvInfer.h HINTS ${TENSOR_DIR} PATH_SUFFIXES include)

add_library(NVCUVID SHARED IMPORTED)
set_property(TARGET NVCUVID PROPERTY IMPORTED_LOCATION "/usr/local/lib/libnvcuvid.so")

add_library(NVENCODE SHARED IMPORTED)
set_property(TARGET NVENCODE PROPERTY IMPORTED_LOCATION "/usr/local/lib/libnvidia-encode.so")

add_subdirectory(include/spdlog)
add_subdirectory(include/indicators)
add_subdirectory(include/highfive)

# Macro for adding targets
function(add_visage_target name sources)
    add_executable(${name} ${sources})
    set_property(TARGET ${name} PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
    target_include_directories(${name} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
target_link_libraries(${name} PRIVATE
    ${OpenCV_LIBS}
    pthread
    dl
    m
    nvinfer
    nvonnxparser
    cudart
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
    spdlog::spdlog 
    indicators
    NVCUVID
    NVENCODE
    HighFive
)
endfunction()

add_compile_options(-Wno-deprecated-declarations)

add_visage_target(visage "src/main.cpp;src/pipeline.cpp;src/utils.cpp;src/image_processor.cpp;src/trt_infer.cpp;src/ops.cpp")
