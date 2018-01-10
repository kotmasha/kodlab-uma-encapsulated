include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)

add_definitions(-DBUILDING_DLL)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB_RECURSE UMAREST
    "*.h"
    "*.cpp"
)

source_group("UMARest" FILES ${UMAREST})
add_library(UMARest SHARED ${UMAREST})
target_link_libraries(UMARest UMAUtil UMAKernel cpprest_2_10.dll)