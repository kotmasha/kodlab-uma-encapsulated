include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)
include_directories(../kernel)

find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)
add_definitions(-DBUILDING_DLL)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(WORLD
    "World.h"
    "World.cpp"
)
set(AGENT
    "Agent.h"
    "Agent.cpp"
)
set(SNAPSHOT
    "Snapshot.h"
    "Snapshot.cpp"
)
set(DATAMANAGER
    "DataManager.h"
    "DataManager.cpp"
)
set(SENSOR
    "Sensor.h"
    "Sensor.cpp"
    "SensorPair.h"
    "SensorPair.cpp"
)
set(GLOB_RECURSE MEASURABLE
    "Measurable.h"
    "Measurable.cpp"
    "MeasurablePair.h"
    "MeasurablePair.cpp"
)
set(SIMULATION
    "Simulation.h"
    "Simulation.cpp"
)
file(GLOB_RECURSE SOURCE
    "*.h"
    "*.cpp"
)

source_group("world" FILES ${WORLD})
source_group("agent" FILES ${AGENT})
source_group("snapshot" FILES ${SNAPSHOT})
source_group("dataManager" FILES ${DATAMANAGER})
source_group("sensor" FILES ${SENSOR})
source_group("measurable" FILES ${MEASURABLE})
source_group("simulation" FILES ${SIMULATION})
cuda_add_library(UMACore SHARED ${SOURCE})
target_link_libraries(UMACore UMAUtil UMAKernel)