#ifndef _SIMULATIONHANDLER_
#define _SIMULATIONHANDLER_

#include "Global.h"
#include "AdminHandler.h"

using namespace std;

class SimulationHandler: public AdminHandler {
public:
	SimulationHandler(string handler_factory);
	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response);
	~SimulationHandler();

protected:
	string_t UMA_DECISION, UMA_UP, UMA_AMPER, UMA_DELAY, UMA_SAVING, UMA_LOADING, UMA_PRUNING, UMA_NPDIRS, UMA_IMPLICATION;
	string_t UMA_SIGNALS, UMA_PHI, UMA_ACTIVE;
	string_t UMA_OBSPLUS, UMA_OBSMINUS;
	string_t UMA_AMPER_LIST, UMA_DELAY_LIST, UMA_UUID_LIST;
	string_t UMA_FILE_NAME;
	string_t UMA_MERGE;
	string_t UMA_SENSORS;
	void create_decision(World *world, http_request &request, http_response &response);
	void create_amper(World *world, http_request &request, http_response &response);
	void create_delay(World *world, http_request &request, http_response &response);
	void create_saving(World *world, json::value &data, http_request &request);
	void create_loading(World *world, json::value &data, http_request &request);
	void create_pruning(World *world, http_request &request, http_response &response);
	void create_merging(World *world, json::value &data, http_request &request);
};

#endif