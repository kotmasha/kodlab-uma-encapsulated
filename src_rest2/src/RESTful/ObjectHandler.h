#ifndef _OBJECTHANDLER_
#define _OBJECTHANDLER_

#include "Global.h"
#include "AdminHandler.h"

using namespace std;

/*
The class will handle all incoming and outcoming request for access data unit
*/
class ObjectHandler : public AdminHandler {
public:
	ObjectHandler(logManager *log_access);
	virtual void handle_create(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_update(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_read(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_delete(World *world, vector<string_t> &paths, http_request &request);
	~ObjectHandler();

protected:
	string_t UMA_INITIAL_SIZE, UMA_THRESHOLD, UMA_Q, UMA_CAL_TARGET;
	string_t UMA_C_SID;
	string_t UMA_AMPER_LIST;
	void create_agent(World *world, json::value &data, http_request &request);
	void create_snapshot(World *world, json::value &data, http_request &request);
	void create_sensor(World *world, json::value &data, http_request &request);

	void update_agent(World *world, json::value &data, http_request &request);
	void update_snapshot(World *world, json::value &data, http_request &request);
	void update_sensor(World *world, json::value &data, http_request &request);
};

#endif