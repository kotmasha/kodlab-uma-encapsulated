include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)

find_package(CUDA REQUIRED)
include(FindCUDA)
set(CUDA_SEPARABLE_COMPILATION ON)
if(UNIX)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
endif(UNIX)

set(DEVICEUTIL
    "device_util.h"
    "device_util.cu"
)
set(KERNELUTIL
    "kernel_util.cuh"
    "kernel_util.cu"
)
set(DATAUTIL
    "data_util.h"
    "data_util.cpp"
)
set(UMABASE
    "uma_base.cuh"
    "uma_base.cu"
)

file(GLOB_RECURSE KERNEL
    "*.h"
    "*.cpp"
    "*.cu"
    "*.cuh"
)


source_group("deviceUtil" FILES ${DEVICEUTIL})
source_group("kernelUtil" FILES ${KERNELUTIL})
source_group("dataUtil" FILES ${DATAUTIL})
source_group("UMABase" FILES ${UMABASE})

cuda_add_library(UMAKernel STATIC ${KERNEL})
target_link_libraries(UMAKernel UMAUtil)
