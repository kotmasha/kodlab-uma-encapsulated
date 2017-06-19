#ifndef _SIMULATIONHANDLER_
#define _SIMULATIONHANDLER_

#include "Global.h"
#include "AdminHandler.h"

using namespace std;

class SimulationHandler: public AdminHandler {
public:
	SimulationHandler(logManager *log_access);
	virtual void handle_create(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_update(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_read(World *world, vector<string_t> &paths, http_request &request);
	virtual void handle_delete(World *world, vector<string_t> &paths, http_request &request);
	~SimulationHandler();

protected:
	string_t UMA_SIGNALS, UMA_PHI, UMA_ACTIVE;
	void create_decision(World *world, json::value &data, http_request &request);
};

#endif