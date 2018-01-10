#ifndef _SENSORHANDLER_
#define _SENSORHANDLER_

#include "Global.h"
#include "UMARestHandler.h"

class SensorHandler : public UMARestHandler {
public:
	SensorHandler(const string &handler_name);
	virtual void handle_create(UMARestRequest &request);
	virtual void handle_update(UMARestRequest &request);
	virtual void handle_read(UMARestRequest &request);
	virtual void handle_delete(UMARestRequest &request);

	void create_sensor(UMARestRequest &request);
	void get_sensor(UMARestRequest &request);
	void get_sensor_pair(UMARestRequest &request);
	void delete_sensor(UMARestRequest &request);
	~SensorHandler();
};

#endif