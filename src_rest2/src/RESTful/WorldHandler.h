#ifndef _WORLDHANDLER_
#define _WORLDHANDLER_

#include "AdminHandler.h"
#include "Global.h"
#include "UMAException.h"

class WorldHandler: public AdminHandler {
public:
	WorldHandler(string handler_factory, logManager *log_access);
	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response);
	void get_world(World *world, http_request &request, http_response &response);
};

#endif
