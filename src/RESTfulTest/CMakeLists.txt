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
set_target_properties(UMARestTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin/test)
set_target_properties(UMARestTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin/test)
set_target_properties(UMARestTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin/test)
