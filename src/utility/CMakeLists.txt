include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(UTILITY
    "UMAUtil.h"
    "UMAUtil.cpp"
)
set(LOGGING
    "Logger.h"
    "Logger.cpp"
)
set(EXCEPTION
    "UMAException.h"
    "UMAException.cpp"
)
set(CONF
    "ConfReader.h"
    "ConfReader.cpp"
)

source_group("utility" FILES ${UTILITY})
source_group("logging" FILES ${LOGGING})
source_group("exception" FILES ${EXCEPTION})
source_group("conf" FILES ${CONF})
add_library(UMAUtil STATIC ${UTILITY} ${LOGGING} ${EXCEPTION} ${CONF})