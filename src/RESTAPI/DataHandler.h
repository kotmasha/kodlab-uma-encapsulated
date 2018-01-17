#ifndef _DATAHANDLER_
#define _DATAHANDLER_

#include "Global.h"
#include "UMARestHandler.h"

using namespace std;

/*
The class will handle all incoming and outcoming request for access data unit
*/
class DataHandler: public UMARestHandler {
public:
	DataHandler(const string &handler_name);
	virtual void handle_create(UMARestRequest &request);
	virtual void handle_update(UMARestRequest &request);
	virtual void handle_read(UMARestRequest &request);
	virtual void handle_delete(UMARestRequest &request);

	~DataHandler();
};

#endif