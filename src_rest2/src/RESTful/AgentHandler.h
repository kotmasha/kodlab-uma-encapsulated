#ifndef _AGENTHANDLER_
#define _AGENTHANDLER_

#include "Global.h"
#include "AdminHandler.h"

class World;
class Agent;

class AgentHandler: public AdminHandler {
public:
	AgentHandler(string handler_factory);
	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response);
	void create_agent(World *world, http_request &request, http_response &response);
	void get_agent(World *world, http_request &request, http_response &response);
	void delete_agent(World *world, http_request &request, http_response &response);
	virtual ~AgentHandler();
};

#endif