include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)
include_directories(../kernel)
include_directories(../core)

file(GLOB_RECURSE CORE_TEST
    "*.h"
    "*.cpp"
)

source_group("CoreTest" FILES ${CORE_TEST})
add_executable(UMACoreTest ${CORE_TEST})

if(MSVC)
    target_link_libraries(UMACoreTest UMACore gtest)
else()
    target_link_libraries(UMACoreTest UMACore gtest pthread)
endif(MSVC)
