#ifndef _WORLDHANDLER_
#define _WORLDHANDLER_

#include "UMARestHandler.h"
#include "UMARestRequest.h"
#include "Global.h"

class WorldHandler: public UMARestHandler {
public:
	WorldHandler(const string &handlerName);
	virtual void handleCreate(UMARestRequest &request);
	virtual void handleUpdate(UMARestRequest &request);
	virtual void handleRead(UMARestRequest &request);
	virtual void handleDelete(UMARestRequest &request);
	void getWorld(UMARestRequest &request);
};

#endif
