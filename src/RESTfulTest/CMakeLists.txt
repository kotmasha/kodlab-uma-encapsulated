include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)
include_directories(../RESTful)

file(GLOB_RECURSE REST_TEST
    "*.h"
    "*.cpp"
)

source_group("RestTest" FILES ${REST_TEST})
add_executable(UMARestTest ${REST_TEST})

if(MSVC)
    target_link_libraries(UMARestTest UMARest gtest)
else()
    target_link_libraries(UMARestTest UMARest gtest pthread)
endif(MSVC)
