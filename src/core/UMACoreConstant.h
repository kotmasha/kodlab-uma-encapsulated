#ifndef _UMACORECONSTANT_
#define _UMACORECONSTANT_

#include "Global.h"
#include "UMAException.h"

namespace UMACoreConstant {
	enum UMA_OBJECT { WORLD, EXPERIMENT, AGENT, SNAPSHOT, DATA_MANAGER, SENSOR, ATTR_SENSOR, SENSOR_PAIR, ATTR_SENSOR_PAIR };
	enum UMA_AGENT { AGENT_STATIONARY, AGENT_QUALITATIVE };
	enum UMA_SNAPSHOT { SNAPSHOT_STATIONARY, SNAPSHOT_QUALITATIVE };

	string getUMAObjName(const UMA_OBJECT &obj);
	string getUMAAgentName(const UMA_AGENT &agent);
	string getUMASnapshotName(const UMA_SNAPSHOT &snapshot);
	UMA_SNAPSHOT getUMASnapshotTypeByAgent(const UMA_AGENT &agent);
}
#endif