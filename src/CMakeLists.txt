cmake_minimum_required(VERSION 2.8)
project(UMA)

if(UNIX)
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
endif(UNIX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)
if(MSVC)
    set(CMAKE_GENERATOR_PLATFORM x64)
    set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
    set(CMAKE_SUPPRESS_REGENERATION ON)
endif(MSVC)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../lib)

if(UNIX)
    message("building on Linux")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC
endif(UNIX)

set(CUDA_PROPAGATE_HOST_FLAGS ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependency/googletest-release-1.8.0/googletest/include)
include_directories(/usr/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependency/build/googletest)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)

add_subdirectory(utility)
if(MSVC)
    add_subdirectory(kernel)
endif(MSVC)
add_subdirectory(core)
add_subdirectory(RESTful)
add_subdirectory(RESTAPI)
add_subdirectory(utilityTest)
add_subdirectory(kernelTest)
add_subdirectory(RESTfulTest)
add_subdirectory(CoreTest)

if(MSVC)
    add_dependencies(UMAKernel UMAUtil)
endif(MSVC)
add_dependencies(UMACore UMAUtil)
add_dependencies(UMARest UMAUtil)
add_dependencies(UMA UMAUtil UMACore UMARest)
add_dependencies(UMAUtilTest UMAUtil)
add_dependencies(UMAKernelTest UMAUtil)
add_dependencies(UMARestTest UMARest)
add_dependencies(UMACoreTest UMACore)