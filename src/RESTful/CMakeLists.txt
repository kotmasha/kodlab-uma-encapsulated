include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)

add_definitions(-DBUILDING_DLL)

file(GLOB_RECURSE UMAREST
    "*.h"
    "*.cpp"
)

source_group("UMARest" FILES ${UMAREST})
add_library(UMARest SHARED ${UMAREST})
if(MSVC)
    target_link_libraries(UMARest UMAUtil cpprest_2_10)
else()
    target_link_libraries(UMARest UMAUtil cpprest crypto ssl boost_system)
endif(MSVC)
