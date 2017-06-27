#ifndef _DATAHANDLER_
#define _DATAHANDLER_

#include "Global.h"
#include "AdminHandler.h"

using namespace std;

/*
The class will handle all incoming and outcoming request for access data unit
*/
class DataHandler: public AdminHandler {
public:
	DataHandler(logManager *log_access);
	virtual void handle_create(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_update(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_read(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_delete(World *world, vector<string_t> &paths, http_request &request);
	~DataHandler();

protected:
	string_t UMA_CURRENT, UMA_PREDICTION, UMA_TARGET;
	string_t UMA_TARGET_LIST;
};

#endif