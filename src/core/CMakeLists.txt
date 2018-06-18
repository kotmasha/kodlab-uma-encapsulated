include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../common)
include_directories(../utility)
include_directories(../kernel)

if(MSVC)
    find_package(CUDA REQUIRED)
    include(FindCUDA)
endif(MSVC)

add_definitions(-DBUILDING_DLL)

set(WORLD
    "World.h"
    "World.cpp"
)
set(EXPERIMENT
    "Experiment.h"
    "Experiment.cpp"
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
set(ATTRSENSOR
    "AttrSensor.h"
    "AttrSensor.cpp"
    "AttrSensorPair.h"
    "AttrSensorPair.cpp"
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
source_group("expriment" FILES ${EXPERIMENT})
source_group("agent" FILES ${AGENT})
source_group("snapshot" FILES ${SNAPSHOT})
source_group("dataManager" FILES ${DATAMANAGER})
source_group("sensor" FILES ${SENSOR})
source_group("attr_sensor" FILES ${ATTRSENSOR})
source_group("simulation" FILES ${SIMULATION})

if(MSVC)
    message("building UMACore under windows using cuda_add_library")
    cuda_add_library(UMACore SHARED ${SOURCE})
    target_link_libraries(UMACore UMAKernel UMAUtil)
else()
    message("building UMACore under Linux using add_library")
    add_library(UMACore SHARED ${SOURCE})
    target_link_libraries(UMACore UMAKernel UMAUtil cuda cudart)
endif()
