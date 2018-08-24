#ifndef _MEASURABLEHANDLER_
#define _MEASURABLEHANDLER_

#include "Global.h"
#include "UMARestHandler.h"

class AttrSensorHandler: public UMARestHandler {
public:
	AttrSensorHandler(const string &handlerName);

	virtual void handleCreate(UMARestRequest &request);
	virtual void handleUpdate(UMARestRequest &request);
	virtual void handleRead(UMARestRequest &request);
	virtual void handleDelete(UMARestRequest &request);

	void getAttrSensor(UMARestRequest &request);
	void getAttrSensorPair(UMARestRequest &request);
	void updateAttrSensorPair(UMARestRequest &request);
	void updateAttrSensor(UMARestRequest &request);
	~AttrSensorHandler();
};

#endif