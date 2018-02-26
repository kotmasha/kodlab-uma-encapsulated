#ifndef _MEASURABLEHANDLER_
#define _MEASURABLEHANDLER_

#include "Global.h"
#include "UMARestHandler.h"

class AttrSensorHandler: public UMARestHandler {
public:
	AttrSensorHandler(const string &handler_name);

	virtual void handle_create(UMARestRequest &request);
	virtual void handle_update(UMARestRequest &request);
	virtual void handle_read(UMARestRequest &request);
	virtual void handle_delete(UMARestRequest &request);

	void get_attr_sensor(UMARestRequest &request);
	void get_attr_sensor_pair(UMARestRequest &request);
	void update_attr_sensor_pair(UMARestRequest &request);
	void update_attr_sensor(UMARestRequest &request);
	~AttrSensorHandler();
};

#endif