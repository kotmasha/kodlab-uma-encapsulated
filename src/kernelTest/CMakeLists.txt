include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)
include_directories(../kernel)

find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

file(GLOB_RECURSE KERNEL_TEST
    "*.h"
    "*.cpp"
)

source_group("kernelTest" FILES ${KERNEL_TEST})
cuda_add_executable(UMAKernelTest ${KERNEL_TEST})
target_link_libraries(UMAKernelTest UMAUtil UMAKernel gtest)
set_target_properties(UMAKernelTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/test )
set_target_properties(UMAKernelTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin/test )
set_target_properties(UMAKernelTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin/test )