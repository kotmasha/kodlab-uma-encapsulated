#ifndef _DATAHANDLER_
#define _DATAHANDLER_

#include "Global.h"
#include "AdminHandler.h"
#include "UMAException.h"

using namespace std;

/*
The class will handle all incoming and outcoming request for access data unit
*/
class DataHandler: public AdminHandler {
public:
	DataHandler(string handler_factory, logManager *log_access);
	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response);

	json::value convert_size_info(const std::map<string, int> &size_info);

	~DataHandler();

protected:
	string_t UMA_CURRENT, UMA_PREDICTION, UMA_TARGET;
	string_t UMA_TARGET_LIST, UMA_SIGNALS, UMA_OBSERVE;
	string_t UMA_DATA_SIZE;
	string_t UMA_WEIGHTS, UMA_DIRS, UMA_THRESHOLDS;
};

#endif