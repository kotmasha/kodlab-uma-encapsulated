#ifndef _WORLD_
#define _WORLD_

#include "Global.h"
using namespace std;

class Experiment;

/*
This is the world class, it maintains all the agent object
*/
class DLL_PUBLIC World{
private:
	static World *_world;
	std::map<string, Experiment*> _experiments;

public:
	static std::map<string, std::map<string, string>> core_info;

public:
	static World *instance();
	Experiment *createExperiment(const string &experimentId);
	Experiment *getExperiment(const string &experimentId);
	void deleteExperiment(const string &experimentId);
	vector<string> getExperimentInfo();

	~World();
};

#endif