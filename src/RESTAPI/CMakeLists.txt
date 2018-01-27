include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)
include_directories(../core)
include_directories(../RESTful)

file(GLOB_RECURSE RESTAPI
    "*.h"
    "*.cpp"
)

source_group("restAPI" FILES ${RESTAPI})
add_executable(UMA ${RESTAPI})
target_link_libraries(UMA UMAUtil UMACore UMARest)
