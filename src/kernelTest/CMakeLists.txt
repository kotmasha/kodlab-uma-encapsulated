include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)
include_directories(../kernel)

if(MSVC)
    find_package(CUDA REQUIRED)
    include(FindCUDA)
endif(MSVC)

file(GLOB_RECURSE KERNEL_TEST
    "*.h"
    "*.cpp"
)

source_group("kernelTest" FILES ${KERNEL_TEST})
if(MSVC)
    message("building UMAKernelTest under windows using cuda_add_executable")
    cuda_add_executable(UMAKernelTest ${KERNEL_TEST})
    target_link_libraries(UMAKernelTest UMAKernel UMAUtil gtest)
else()
    message("building UMAKernelTest under Linux using add_executable")
    add_executable(UMAKernelTest ${KERNEL_TEST})
    target_link_libraries(UMAKernelTest UMAKernel UMAUtil gtest cuda cudart pthread)
endif(MSVC)
set_target_properties(UMAKernelTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/test)
set_target_properties(UMAKernelTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin/test)
set_target_properties(UMAKernelTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin/test)
