#ifndef _AGENTHANDLER_
#define _AGENTHANDLER_

#include "Global.h"
#include "UMARestHandler.h"
#include "UMARestRequest.h"

class World;
class Agent;

class AgentHandler: public UMARestHandler {
public:
	AgentHandler(const string &handler_name);
	virtual void handle_create(UMARestRequest &request);
	virtual void handle_update(UMARestRequest &request);
	virtual void handle_read(UMARestRequest &request);
	virtual void handle_delete(UMARestRequest &request);
	void create_agent(UMARestRequest &request);
	void get_agent(UMARestRequest &request);
	void delete_agent(UMARestRequest &request);
	virtual ~AgentHandler();
};

#endif