cmake_minimum_required(VERSION 2.8)
project(UMA)

set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_SUPPRESS_REGENERATION true)

add_subdirectory(utility)
add_subdirectory(kernel)
add_subdirectory(core)
add_subdirectory(RESTful)
add_subdirectory(RESTAPI)

add_dependencies(UMAKernel UMAUtil)
add_dependencies(UMACore UMAUtil UMAKernel)
add_dependencies(UMARest UMAUtil)
add_dependencies(UMA UMAUtil UMACore UMARest)