#ifndef _WORLDHANDLER_
#define _WORLDHANDLER_

#include "UMARestHandler.h"
#include "UMARestRequest.h"
#include "Global.h"

class WorldHandler: public UMARestHandler {
public:
	WorldHandler(const string &handler_name);
	virtual void handle_create(UMARestRequest &request);
	virtual void handle_update(UMARestRequest &request);
	virtual void handle_read(UMARestRequest &request);
	virtual void handle_delete(UMARestRequest &request);
	void get_world(UMARestRequest &request);
};

#endif
