include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)
include_directories(../core)
include_directories(../RESTful)

file(GLOB_RECURSE LAUNCHER
    "*.h"
    "*.c"
)

source_group("launcher" FILES ${LAUNCHER})
add_executable(UMA ${LAUNCHER})
#target_link_libraries(UMA UMAUtil UMACore UMARest)