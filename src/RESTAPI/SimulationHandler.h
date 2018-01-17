#ifndef _SIMULATIONHANDLER_
#define _SIMULATIONHANDLER_

#include "Global.h"
#include "UMARestHandler.h"

using namespace std;

class SimulationHandler: public UMARestHandler {
public:
	SimulationHandler(const string &handler_name);
	virtual void handle_create(UMARestRequest &request);
	virtual void handle_update(UMARestRequest &request);
	virtual void handle_read(UMARestRequest &request);
	virtual void handle_delete(UMARestRequest &request);
	~SimulationHandler();

protected:
	void create_decision(UMARestRequest &request);
	void create_propagation(UMARestRequest &request);
	void create_npdirs(UMARestRequest &request);
	void create_up(UMARestRequest &request);
	void create_ups(UMARestRequest &request);
	void create_downs(UMARestRequest &request);
	void create_blocks(UMARestRequest &request);
	void create_abduction(UMARestRequest &request);
	void create_propagate_masks(UMARestRequest &request);
	//void create_saving(World *world, json::value &data, http_request &request);
	//void create_loading(World *world, json::value &data, http_request &request);
	//void create_merging(World *world, json::value &data, http_request &request);
};

#endif