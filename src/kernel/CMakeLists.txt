include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)

find_package(CUDA REQUIRED)
include(FindCUDA)
set(CUDA_SEPARABLE_COMPILATION ON)
if(UNIX)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
endif(UNIX)

file(GLOB_RECURSE KERNEL
    "*.h"
    "*.cpp"
    "*.cu"
    "*.cuh"
)

cuda_add_library(UMAKernel STATIC ${KERNEL})
target_link_libraries(UMAKernel UMAUtil)
