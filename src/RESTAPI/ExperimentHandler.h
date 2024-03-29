#ifndef _EXPERIMENTHANDLER_
#define _EXPERIMENTHANDLER_

#include "Global.h"
#include "UMARestHandler.h"
#include "UMARestRequest.h"

class World;
class Agent;

class ExperimentHandler : public UMARestHandler {
public:
	ExperimentHandler(const string &handlerName);
	virtual void handleCreate(UMARestRequest &request);
	virtual void handleUpdate(UMARestRequest &request);
	virtual void handleRead(UMARestRequest &request);
	virtual void handleDelete(UMARestRequest &request);
	void createExperiment(UMARestRequest &request);
	void getExperiment(UMARestRequest &request);
	void deleteExperiment(UMARestRequest &request);
	void saveExperiment(UMARestRequest &request);
	void loadExperiment(UMARestRequest &request);
	virtual ~ExperimentHandler();
};

#endif