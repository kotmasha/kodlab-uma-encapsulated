#ifndef _SENSORHANDLER_
#define _SENSORHANDLER_

#include "Global.h"
#include "UMARestHandler.h"

class SensorHandler : public UMARestHandler {
public:
	SensorHandler(const string &handler_name);
	virtual void handleCreate(UMARestRequest &request);
	virtual void handleUpdate(UMARestRequest &request);
	virtual void handleRead(UMARestRequest &request);
	virtual void handleDelete(UMARestRequest &request);

	void createSensor(UMARestRequest &request);
	void getSensor(UMARestRequest &request);
	void getSensorPair(UMARestRequest &request);
	void deleteSensor(UMARestRequest &request);
	~SensorHandler();
};

#endif