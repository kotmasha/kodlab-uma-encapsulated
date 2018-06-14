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
	virtual void handleCreate(UMARestRequest &request);
	virtual void handleUpdate(UMARestRequest &request);
	virtual void handleRead(UMARestRequest &request);
	virtual void handleDelete(UMARestRequest &request);

	~DataHandler();
};

#endif