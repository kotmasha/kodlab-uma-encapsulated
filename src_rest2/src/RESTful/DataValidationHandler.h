#ifndef _DATAVALIDATIONHANDLER_
#define _DATAVALIDATIONHANDLER_

#include "Global.h"
#include "AdminHandler.h"

using namespace std;

class DataValidationHandler : public AdminHandler {
public:
	DataValidationHandler(logManager *log_access);
	virtual void handle_create(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_update(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_read(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_delete(World *world, vector<string_t> &paths, http_request &request);
	
	~DataValidationHandler();

protected:
	logManager *_log_access;
	string_t UMA_INITIAL_SIZE;
	void validate_snapshot(World *world, json::value &data, http_request &request);
};

#endif
