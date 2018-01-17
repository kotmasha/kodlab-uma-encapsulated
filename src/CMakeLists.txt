cmake_minimum_required(VERSION 2.8)
project(UMA)

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)

set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_SUPPRESS_REGENERATION true)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependency/googletest-release-1.8.0/googletest/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependency/build/googletest/Release)

add_subdirectory(utility)
add_subdirectory(kernel)
add_subdirectory(core)
add_subdirectory(RESTful)
add_subdirectory(RESTAPI)
add_subdirectory(utilityTest)
add_subdirectory(kernelTest)

add_dependencies(UMAKernel UMAUtil)
add_dependencies(UMACore UMAUtil UMAKernel)
add_dependencies(UMARest UMAUtil)
add_dependencies(UMA UMAUtil UMACore UMARest)
add_dependencies(UMAUtilTest UMAUtil)
add_dependencies(UMAKernelTest UMAUtil UMAKernel)