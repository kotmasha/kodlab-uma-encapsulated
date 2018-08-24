#ifndef _AGENTHANDLER_
#define _AGENTHANDLER_

#include "Global.h"
#include "UMARestHandler.h"
#include "UMARestRequest.h"

class World;
class Agent;

class AgentHandler: public UMARestHandler {
public:
	AgentHandler(const string &handlerName);
	virtual void handleCreate(UMARestRequest &request);
	virtual void handleUpdate(UMARestRequest &request);
	virtual void handleRead(UMARestRequest &request);
	virtual void handleDelete(UMARestRequest &request);
	void createAgent(UMARestRequest &request);
	void getAgent(UMARestRequest &request);
	void deleteAgent(UMARestRequest &request);
	virtual ~AgentHandler();
};

#endif