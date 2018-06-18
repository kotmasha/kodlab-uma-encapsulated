#include "SimulationHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "Simulation.h"
#include "UMAException.h"

SimulationHandler::SimulationHandler(const string &handler_name): UMARestHandler(handler_name) {
}

void SimulationHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/simulation/decision") {
		createDecision(request);
		return;
	}
	else if (requestUrl == "/UMA/simulation/up") {
		createUp(request);
		return;
	}
	else if (requestUrl == "/UMA/simulation/ups") {
		createUps(request);
		return;
	}
	else if (requestUrl == "/UMA/simulation/propagation") {
		createPropagation(request);
		return;
	}
	else if (requestUrl == "/UMA/simulation/npdirs") {
		createNpdirs(request);
		return;
	}
	else if (requestUrl == "/UMA/simulation/downs") {
		createDowns(request);
		return;
	}
	else if (requestUrl == "/UMA/simulation/blocks") {
		createBlocks(request);
		return;
	}
	else if (requestUrl == "/UMA/simulation/abduction") {
		createAbduction(request);
		return;
	}
	else if (requestUrl == "/UMA/simulation/propagateMasks") {
		createPropagateMasks(request);
		return;
	}

	throw UMAException("Cannot handle POST " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SimulationHandler::handleUpdate(UMARestRequest &request) {
	throw UMAException("Cannot handle PUT " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SimulationHandler::handleRead(UMARestRequest &request) {
	throw UMAException("Cannot handle GET " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SimulationHandler::handleDelete(UMARestRequest &request) {
	throw UMAException("Cannot handle DELETE " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SimulationHandler::createDecision(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const double phi = request.get_double_data("phi");
	const bool active = request.get_bool_data("active");
	vector<bool> obsPlus = request.get_bool1d_data("obs_plus");
	vector<bool> obsMinus = request.get_bool1d_data("obs_minus");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshotPlus = agent->getSnapshot("plus");
	Snapshot *snapshotMinus = agent->getSnapshot("minus");
	vector<float> res = simulation::decide(agent, obsPlus, obsMinus, phi, active);

	vector<vector<bool>> current, prediction, target;
	current.push_back(snapshotPlus->getDM()->getCurrent());
	current.push_back(snapshotMinus->getDM()->getCurrent());
	prediction.push_back(snapshotPlus->getDM()->getPrediction());
	prediction.push_back(snapshotMinus->getDM()->getPrediction());
	target.push_back(snapshotPlus->getDM()->getTarget());
	target.push_back(snapshotMinus->getDM()->getTarget());

	request.set_data("res_plus", res[0]);
	request.set_data("res_minus", res[1]);
	request.set_data("current_plus", current[0]);
	request.set_data("current_minus", current[1]);
	request.set_data("prediction_plus", prediction[0]);
	request.set_data("prediction_minus", prediction[1]);
	request.set_data("target_plus", target[0]);
	request.set_data("target_minus", target[1]);
	request.set_message("decision made for snapshots");
}

/*
void SimulationHandler::create_saving(World *world, json::value &data, http_request &request) {
	string name;
	try {
		name = get_string_input(data, UMA_FILE_NAME, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	try {
		world->save_world(name);
		json::value message;
		message[MESSAGE] = json::value::string(U("data saved successfully to ") + data[UMA_FILE_NAME].as_string() + U(".uma"));
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[MESSAGE] = json::value::string(U("cannot saving data!"));
		request.reply(status_codes::BadRequest, message);
	}
}

void SimulationHandler::create_loading(World *world, json::value &data, http_request &request) {
	string name;
	try {
		name = get_string_input(data, UMA_FILE_NAME, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	try {
		world->load_world(name);
		json::value message;
		message[MESSAGE] = json::value::string(U("data loaded successfully loaded from ") + data[UMA_FILE_NAME].as_string() + U(".uma"));
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[MESSAGE] = json::value::string(U("cannot load data!"));
		request.reply(status_codes::BadRequest, message);
	}
}
*/

/*
void SimulationHandler::create_merging(World *world, json::value &data, http_request &request) {
try {
world->merge_test();
json::value message;

message[MESSAGE] = json::value::string(U("test data merged successfully"));
_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
request.reply(status_codes::Created, message);
return;
}
catch (exception &e) {
_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
json::value message;
message[MESSAGE] = json::value::string(U("test data merged error!"));
request.reply(status_codes::BadRequest, message);
}
}

void SimulationHandler::create_implication(World *world, json::value &data, http_request &request) {
string agent_id, snapshot_id;
vector<string> sensors;
try {
agent_id = get_string_input(data, UMA_AGENT_ID, request);
snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
sensors = get_string1d_input(data, UMA_SENSORS, request);
}
catch (exception &e) {
cout << e.what() << endl;
return;
}

Agent *agent = NULL;
Snapshot *snapshot = NULL;
if (!get_agent_by_id(world, agent_id, agent, request)) return;
if (!get_snapshot_by_id(agent, snapshot_id, snapshot, request)) return;
try {
snapshot->create_implication(sensors[0], sensors[1]);
json::value message;

message[MESSAGE] = json::value::string(U("implication made successful"));
_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
request.reply(status_codes::Created, message);
return;
}
catch (exception &e) {
_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
json::value message;
message[MESSAGE] = json::value::string(U("implication made error!"));
request.reply(status_codes::BadRequest, message);
}
}
*/


void SimulationHandler::createUp(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const vector<bool> signal = request.get_bool1d_data("signal");
	vector<vector<bool> > signals(1, signal);

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> sizeInfo = dm->getSizeInfo();
	int attrSensorSize = sizeInfo["_attr_sensor_size"];
	dm->setSignals(signals);
	simulation::ups_GPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, 1, attrSensorSize);

	vector<vector<bool> > results = dm->getSignals(1);

	request.set_message("Up created");
	request.set_data("signal", results[0]);
}

void SimulationHandler::createUps(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const vector<vector<bool> > signals = request.get_bool2d_data("signals");
	int sig_count = signals.size();

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> sizeInfo = dm->getSizeInfo();
	int attrSensorSize = sizeInfo["_attr_sensor_size"];

	dm->setSignals(signals);
	simulation::ups_GPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, sig_count, attrSensorSize);

	vector<vector<bool> > results = dm->getSignals(signals.size());

	request.set_message("Ups created");
	request.set_data("signals", results);
}

void SimulationHandler::createDowns(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const vector<vector<bool> > signals = request.get_bool2d_data("signals");
	int sigCount = signals.size();

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> sizeInfo = dm->getSizeInfo();
	int attrSensorSize = sizeInfo["_attr_sensor_size"];

	dm->setSignals(signals);
	simulation::downs_GPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, sigCount, attrSensorSize);

	vector<vector<bool> > results = dm->getSignals(signals.size());

	request.set_message("Downs created");
	request.set_data("signals", results);
}

void SimulationHandler::createPropagation(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const vector<vector<bool> > signals = request.get_bool2d_data("signals");
	const vector<bool> load = request.get_bool1d_data("load");
	int sigCount = signals.size();

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> sizeInfo = dm->getSizeInfo();
	int attrSensorSize = sizeInfo["_attr_sensor_size"];

	dm->setSignals(signals);
	dm->setLoad(load);

	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), NULL, sigCount, attrSensorSize);
	vector<vector<bool> > results = dm->getLSignals(signals.size());

	request.set_message("Propagation created");
	request.set_data("signals", results);
}

void SimulationHandler::createNpdirs(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	simulation::floyd(dm);
	vector<bool> npdirs = dm->getNPDir();

	request.set_message("N power dir created");
	request.set_data("npdirs", npdirs);
}

void SimulationHandler::createBlocks(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const double delta = request.get_double_data("delta");
	const vector<vector<int> > dists = request.get_int2d_data("dists");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	dm->setDists(dists);
	vector<vector<int> > results = simulation::blocks_GPU(dm, delta);
	
	request.set_message("block GPU created");
	request.set_data("blocks", results);
}

void SimulationHandler::createAbduction(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const vector<vector<bool> > signals = request.get_bool2d_data("signals");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	vector<vector<vector<bool> > > results = simulation::abduction(dm, signals);

	request.set_message("abduction GPU created");
	request.set_data("abduction_even", results[0]);
	request.set_data("abduction_odd", results[1]);
}

void SimulationHandler::createPropagateMasks(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	simulation::propagate_mask(dm);
	vector<vector<bool> > results = dm->getNpdirMasks();

	request.set_message("Mask propagated");
	request.set_data("propagated_mask", results);
}

SimulationHandler::~SimulationHandler() {}