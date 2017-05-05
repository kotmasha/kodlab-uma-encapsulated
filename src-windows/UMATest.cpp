#include "UMATest.h"
#include "Agent.h"

extern int ind(int row, int col);
extern int compi(int x);

CPUTest::CPUTest(){}

CPUTest::~CPUTest(){}

GPUTest::GPUTest(){}

GPUTest::~GPUTest(){}


/*
The below code is unit test code for kernel.cu
*/
/*
-----------------CPU TEST----------------------
*/

int CPUTest::TEST_ind_host(int row, int col){
	return ind(row, col);
}
//ind host test

int CPUTest::TEST_compi_host(int x){
	return compi(x);
}
//compi host test

vector<bool> CPUTest::TEST_up_GPU(vector<bool> signal, vector<bool> dir){
	Agent *agent = new Agent(Agent::FORGETFUL, false);
	vector<string> names;
	agent->init_data("test", signal.size() / 2, names, "");
	agent->copy_dir(0, dir);

	agent->up_GPU(signal, false);

	vector<bool> results = agent->getUp();

	delete agent;

	return results;
}

vector<bool> CPUTest::TEST_gen_mask(vector<bool> mask_amper, vector<bool> current, int base_sensor_size){
	Agent *agent = new Agent(Agent::FORGETFUL, false);
	vector<string> names;
	agent->init_data("test", current.size() / 2, names, "");
	
	agent->copy_mask_amper(0, mask_amper);
	agent->copy_current(0, current);
	agent->reset_base_sensor_size(base_sensor_size, 0);

	agent->gen_mask();

	vector<bool> result = agent->getMask();

	delete agent;

	return result;
}

vector<bool> CPUTest::TEST_set_signal(vector<bool> signal){
	Agent *agent = new Agent(Agent::FORGETFUL, false);
	vector<string> names;
	agent->init_data("test", signal.size() / 2, names, "");
	agent->setSignal(signal);
	
	vector<bool> result = agent->getObserve();

	delete agent;

	return result;
}

vector<double> CPUTest::TEST_init_weight(int sensor_size){
	Agent *agent = new Agent(Agent::FORGETFUL, false);
	vector<string> names;
	agent->init_data("test", sensor_size, names, "");

	vector<double> result = agent->getWeight();

	delete agent;
	return result;
}

vector<bool> CPUTest::TEST_init_direction(int sensor_size){
	Agent *agent = new Agent(Agent::FORGETFUL, false);
	vector<string> names;
	agent->init_data("test", sensor_size, names, "");
	
	vector<bool> result = agent->getDir();

	delete agent;
	return result;
}

vector<bool> CPUTest::TEST_init_mask_amper(int sensor_size){
	Agent *agent = new Agent(Agent::FORGETFUL, false);
	vector<string> names;
	agent->init_data("test", sensor_size, names, "");

	vector<bool> result = agent->getMask_amper();

	delete agent;
	return result;
}

vector<double> CPUTest::TEST_delay(vector<vector<double> > weights, vector<double> measurable, vector<double> measurable_old, double last_total, int measurable_id){
	Agent *agent = new Agent(Agent::FORGETFUL, false);
	vector<string> names;
	agent->init_data("test", measurable.size() / 2, names, "");

	vector<vector<bool> > input;
	vector<bool> tmp;
	for(int i = 0; i < measurable.size(); ++i) tmp.push_back(i == measurable_id);
	input.push_back(tmp);

	vector<vector<bool> > mask_amper = agent->getMask_amper2D();

	agent->copy_weight(0, 0, measurable.size(), measurable, measurable_old, weights, mask_amper);
	agent->setLastTotal(last_total);
	agent->delay(input);
	
	vector<vector<double> > weights2D = agent->getWeight2D();
	vector<double> results = weights2D[weights2D.size() - 2];
	vector<double> weight = weights2D.back();
	results.insert(results.end(), weight.begin(), weight.end());
	return results;
}

/*
-----------------------CPU TEST---------------------
*/
