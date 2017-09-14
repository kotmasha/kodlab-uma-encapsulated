#ifndef _SENSORHANDLER_
#define _SENSORHANDLER_

#include "Global.h"
#include "AdminHandler.h"

class SensorHandler : public AdminHandler {
protected:
	string_t UMA_THRESHOLD, UMA_Q, UMA_AUTO_TARGET;
	string_t UMA_C_SID;
	string_t UMA_AMPER_LIST;
	string_t UMA_SENSOR1, UMA_SENSOR2;
public:
	SensorHandler(string handler_factory, logManager *log_access);
	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response);

	void create_sensor(World *world, http_request &request, http_response &response);
	void get_sensor(World *world, http_request &request, http_response &response);
	void get_sensor_pair(World *world, http_request &request, http_response &response);
	void delete_sensor(World *world, http_request &request, http_response &response);
	~SensorHandler();
};

#endif