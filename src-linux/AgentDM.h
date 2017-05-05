#ifndef _AGENTDM_
#define _AGENTDM_
#include "Global.h"
#include <fstream>
using namespace std;

class Agent;

class AgentDM{
protected:
	Agent *agent;
public:
	AgentDM(Agent *agent);
	~AgentDM();

	bool writeData(string filename);
	bool readData(string filename);
};

#endif