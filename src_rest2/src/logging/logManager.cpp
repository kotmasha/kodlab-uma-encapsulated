#include "LogManager.h"
#include "UMAutil.h"
#include "Logger.h"
#include "ConfReader.h"

Logger accessLogger;
Logger serverLogger;
Logger worldLogger;
Logger agentLogger;
Logger snapshotLogger;
Logger sensorLogger;
Logger sensorPairLogger;
Logger measurableLogger;
Logger measurablePairLogger;
Logger dataManagerLogger;
Logger simulationLogger;

LogManager::LogManager() {
	_log_cfg = ConfReader::read_log_configure();
}

void LogManager::init_logger() {
	accessLogger = Logger("Access", "UMAC_access.log", string_to_log_level(_log_cfg["Access"]["level"]));
	serverLogger = Logger("Server", "UMA_server.log", string_to_log_level(_log_cfg["Server"]["level"]));
	worldLogger = Logger("World", "world.log", string_to_log_level(_log_cfg["World"]["level"]));
	agentLogger = Logger("Agent", "agent.log", string_to_log_level(_log_cfg["Agent"]["level"]));
	snapshotLogger = Logger("Snapshot", "snapshot.log", string_to_log_level(_log_cfg["Snapshot"]["level"]));
	sensorLogger = Logger("Sensor", "sensor.log", string_to_log_level(_log_cfg["Sensor"]["level"]));
	sensorPairLogger = Logger("SensorPair", "sensor.log", string_to_log_level(_log_cfg["SensorPair"]["level"]));
	measurableLogger = Logger("Measurable", "measurable.log", string_to_log_level(_log_cfg["Measurable"]["level"]));
	measurablePairLogger = Logger("MeasurablePair", "measurable.log", string_to_log_level(_log_cfg["MeasurablePair"]["level"]));
	dataManagerLogger = Logger("DataManager", "dataManager.log", string_to_log_level(_log_cfg["DataManager"]["level"]));
	simulationLogger = Logger("Simulation", "simulation.log", string_to_log_level(_log_cfg["Simulation"]["level"]));

	serverLogger.info("Server logger component created");
	serverLogger.info("Access logger component created");
	serverLogger.info("World logger component created");
	serverLogger.info("Agent logger component created");
	serverLogger.info("Snapshot logger component created");
	serverLogger.info("Sensor logger component created");
	serverLogger.info("SensorPair logger component created");
	serverLogger.info("Measurable logger component created");
	serverLogger.info("MeasurablePair logger component created");
	serverLogger.info("DataManager logger component created");
	serverLogger.info("Simulation logger component created");
}

void LogManager::init_log_dirs() {
	UMA_mkdir("log");
}