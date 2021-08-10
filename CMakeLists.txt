cmake_minimum_required(VERSION 2.8)
project(yoloxTracker)

## 基础配置
# list(APPEND CMAKE_MODULE_PATH /usr/local/share/OpenCV)
ADD_DEFINITIONS("-std=c++14 -w -O2 -pthread")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE debug)

## OpenMP相关配置
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


## OpenCV相关设置
find_package(OpenCV 3.4)
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${OpenCV_DIR}/include)

## CUDA及TensorRT相关
find_package(CUDA REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include)
# cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)
# cudnn
include_directories(/home/chen/cuda/include)
link_directories(/home/chen/cuda/lib64)
# tensorrt
include_directories(/home/chen/DeepLearning/settings/TensorRT-7.2.1.6/include)
link_directories(/home/chen/DeepLearning/settings/TensorRT-7.2.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

## 本文件配置
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/src/YOLOX
${PROJECT_SOURCE_DIR}/src/KCF-DSST
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/YOLOX/yolox.cc
src/KCF-DSST/fhog.cc
src/KCF-DSST/json.cc
src/KCF-DSST/kcftracker.cc
src/KCF-DSST/objecttracking.cc
src/DataAssociation.cc
)

target_link_libraries(${PROJECT_NAME}
OpenMP::OpenMP_CXX
nvinfer
cudart
${OpenCV_LIBS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(yolox_tracker ${PROJECT_SOURCE_DIR}/example/YOLOX_Tracker.cpp)
target_link_libraries(yolox_tracker ${PROJECT_NAME})



