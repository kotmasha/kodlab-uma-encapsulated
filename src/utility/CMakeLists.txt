include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)

set(UTILITY
    "UMAutil.h"
    "UMAutil.cpp"
)
set(LOGGING
    "Logger.h"
    "Logger.cpp"
    "LogService.h"
    "LogService.cpp"
)
set(EXCEPTION
    "UMAException.h"
    "UMAException.cpp"
)
set(CONF
    "ConfReader.h"
    "ConfReader.cpp"
)
set(PROPERTY
    "PropertyMap.h"
    "PropertyMap.cpp"
    "PropertyPage.h"
    "PropertyPage.cpp"
)

source_group("utility" FILES ${UTILITY})
source_group("logging" FILES ${LOGGING})
source_group("exception" FILES ${EXCEPTION})
source_group("conf" FILES ${CONF})
source_group("property" FILES ${PROPERTY})
add_library(UMAUtil STATIC ${UTILITY} ${LOGGING} ${EXCEPTION} ${CONF} ${PROPERTY})
