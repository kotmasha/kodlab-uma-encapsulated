#include "UMACoreConstant.h"

string UMACoreConstant::getUMAObjName(const UMA_OBJECT &obj) {
	switch (obj) {
	case UMA_OBJECT::WORLD: return "World";
	case UMA_OBJECT::EXPERIMENT: return "Experiment";
	case UMA_OBJECT::AGENT: return "Agent";
	case UMA_OBJECT::SNAPSHOT: return "Snapshot";
	case UMA_OBJECT::DATA_MANAGER: return "DataManager";
	case UMA_OBJECT::SENSOR: return "Sensor";
	case UMA_OBJECT::SENSOR_PAIR: return "SensorPair";
	case UMA_OBJECT::ATTR_SENSOR: return "AttributeSensor";
	case UMA_OBJECT::ATTR_SENSOR_PAIR: return "AttributeSensorPair";
	default: throw UMAInvalidArgsException("Cannot find the UMA object type=" + to_string(obj));
	}
}

string UMACoreConstant::getUMAAgentName(const UMA_AGENT &agent) {
	switch (agent) {
	case UMA_AGENT::AGENT_STATIONARY: return "Stationary";
	case UMA_AGENT::AGENT_QUALITATIVE: return "Qualitative";
	case UMA_AGENT::AGENT_DISCOUNTED: return "Discounted";
	case UMA_AGENT::AGENT_EMPIRICAL: return "Empirical";
	default: throw UMAInvalidArgsException("Cannot find the UMA agent type=" + to_string(agent));
	}
}

string UMACoreConstant::getUMASnapshotName(const UMA_SNAPSHOT &snapshot) {
	switch (snapshot) {
	case UMA_SNAPSHOT::SNAPSHOT_STATIONARY: return "Stationary";
	case UMA_SNAPSHOT::SNAPSHOT_QUALITATIVE: return "Qualitative";
	case UMA_SNAPSHOT::SNAPSHOT_DISCOUNTED: return "Discounted";
	case UMA_SNAPSHOT::SNAPSHOT_EMPIRICAL: return "Empirical";
	default: throw UMAInvalidArgsException("Cannot find the UMA snapshot type=" + to_string(snapshot));
	}
}

UMACoreConstant::UMA_SNAPSHOT UMACoreConstant::getUMASnapshotTypeByAgent(const UMA_AGENT &agent){
	switch (agent) {
	case UMA_AGENT::AGENT_STATIONARY: return UMA_SNAPSHOT::SNAPSHOT_STATIONARY;
	case UMA_AGENT::AGENT_QUALITATIVE: return UMA_SNAPSHOT::SNAPSHOT_QUALITATIVE;
	case UMA_AGENT::AGENT_DISCOUNTED: return UMA_SNAPSHOT::SNAPSHOT_DISCOUNTED;
	case UMA_AGENT::AGENT_EMPIRICAL: return UMA_SNAPSHOT::SNAPSHOT_EMPIRICAL;
	default: throw UMAInvalidArgsException("Invalid agent type, type=" + to_string(agent));
	}
}