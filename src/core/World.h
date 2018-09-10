#ifndef _WORLD_
#define _WORLD_

#include "Global.h"
#include "UMACoreObject.h"
using namespace std;

class Experiment;

/*
This is the world class, it maintains all the agent object
*/
class DLL_PUBLIC World: public UMACoreObject{
private:
	// the singleton object
	static World *_world;
	// the map for all experiments
	std::map<string, Experiment*> _experiments;

public:
	World();
	static World *instance();
	Experiment *createExperiment(const string &experimentId);
	Experiment *getExperiment(const string &experimentId);
	void deleteExperiment(const string &experimentId);
	vector<string> getExperimentInfo();

	~World();
};

#endif