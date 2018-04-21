include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)

file(GLOB_RECURSE UTILITY_TEST
    "*.h"
    "*.cpp"
)

source_group("utilTest" FILES ${UTILITY_TEST})
add_executable(UMAUtilTest ${UTILITY_TEST})

if(MSVC)
    target_link_libraries(UMAUtilTest UMAUtil gtest)
else()
    target_link_libraries(UMAUtilTest UMAUtil gtest pthread)
endif(MSVC)
