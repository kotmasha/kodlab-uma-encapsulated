#ifndef _MEASURABLEHANDLER_
#define _MEASURABLEHANDLER_

#include "Global.h"
#include "AdminHandler.h"

class MeasurableHandler: public AdminHandler {
public:
	MeasurableHandler(string handler_factory, logManager *log_access);

	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response);

	void get_measurable(World *world, http_request &request, http_response &response);
	void get_measurable_pair(World *world, http_request &request, http_response &response);
	void update_measurable_pair(World *world, http_request &request, http_response &response);
	void update_measurable(World *world, http_request &request, http_response &response);
	~MeasurableHandler();

protected:
	string_t UMA_MEASURABLE1, UMA_MEASURABLE2;
	string_t UMA_W, UMA_D;
	string_t UMA_DIAG, UMA_OLD_DIAG;
};

#endif