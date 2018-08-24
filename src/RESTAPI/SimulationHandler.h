#ifndef _SIMULATIONHANDLER_
#define _SIMULATIONHANDLER_

#include "Global.h"
#include "UMARestHandler.h"

using namespace std;

class SimulationHandler: public UMARestHandler {
public:
	SimulationHandler(const string &handlerName);
	virtual void handleCreate(UMARestRequest &request);
	virtual void handleUpdate(UMARestRequest &request);
	virtual void handleRead(UMARestRequest &request);
	virtual void handleDelete(UMARestRequest &request);
	~SimulationHandler();

protected:
	void createDecision(UMARestRequest &request);
	void createPropagation(UMARestRequest &request);
	void createNpdirs(UMARestRequest &request);
	void createUp(UMARestRequest &request);
	void createUps(UMARestRequest &request);
	void createDowns(UMARestRequest &request);
	void createBlocks(UMARestRequest &request);
	void createAbduction(UMARestRequest &request);
	void createPropagateMasks(UMARestRequest &request);
	//void create_saving(World *world, json::value &data, http_request &request);
	//void create_loading(World *world, json::value &data, http_request &request);
	//void create_merging(World *world, json::value &data, http_request &request);
};

#endif