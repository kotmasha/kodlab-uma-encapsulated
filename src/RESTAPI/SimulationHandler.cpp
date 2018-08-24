#include "SimulationHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "Simulation.h"
#include "UMAException.h"

static Logger serverLogger("Server", "log/UMA_server.log");
SimulationHandler::SimulationHandler(const string &handlerName): UMARestHandler(handlerName) {
}

void SimulationHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
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

	throw UMABadOperationException("Cannot handle POST " + request.getRequestUrl(), false, &serverLogger);
}

void SimulationHandler::handleUpdate(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle PUT " + request.getRequestUrl(), false, &serverLogger);
}

void SimulationHandler::handleRead(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle GET " + request.getRequestUrl(), false, &serverLogger);
}

void SimulationHandler::handleDelete(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle DELETE " + request.getRequestUrl(), false, &serverLogger);
}

void SimulationHandler::createDecision(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const double phi = request.getDoubleData("phi");
	const bool active = request.getBoolData("active");
	vector<bool> obsPlus = request.getBool1dData("obs_plus");
	vector<bool> obsMinus = request.getBool1dData("obs_minus");

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

	request.setData("res_plus", res[0]);
	request.setData("res_minus", res[1]);
	request.setData("current_plus", current[0]);
	request.setData("current_minus", current[1]);
	request.setData("prediction_plus", prediction[0]);
	request.setData("prediction_minus", prediction[1]);
	request.setData("target_plus", target[0]);
	request.setData("target_minus", target[1]);
	request.setMessage("decision made for snapshots");
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
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const vector<bool> signal = request.getBool1dData("signal");
	vector<vector<bool> > signals(1, signal);

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> sizeInfo = dm->getSizeInfo();
	int attrSensorSize = sizeInfo["_attrSensorSize"];
	dm->setSignals(signals);
	simulation::upsGPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, 1, attrSensorSize);

	vector<vector<bool> > results = dm->getSignals(1);

	request.setMessage("Up created");
	request.setData("signal", results[0]);
}

void SimulationHandler::createUps(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const vector<vector<bool> > signals = request.getBool2dData("signals");
	int sigCount = signals.size();

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> sizeInfo = dm->getSizeInfo();
	int attrSensorSize = sizeInfo["_attrSensorSize"];

	dm->setSignals(signals);
	simulation::upsGPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, sigCount, attrSensorSize);

	vector<vector<bool> > results = dm->getSignals(signals.size());

	request.setMessage("Ups created");
	request.setData("signals", results);
}

void SimulationHandler::createDowns(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const vector<vector<bool> > signals = request.getBool2dData("signals");
	int sigCount = signals.size();

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> sizeInfo = dm->getSizeInfo();
	int attrSensorSize = sizeInfo["_attrSensorSize"];

	dm->setSignals(signals);
	simulation::downsGPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, sigCount, attrSensorSize);

	vector<vector<bool> > results = dm->getSignals(signals.size());

	request.setMessage("Downs created");
	request.setData("signals", results);
}

void SimulationHandler::createPropagation(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const vector<vector<bool> > signals = request.getBool2dData("signals");
	const vector<bool> load = request.getBool1dData("load");
	int sigCount = signals.size();

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> sizeInfo = dm->getSizeInfo();
	int attrSensorSize = sizeInfo["_attrSensorSize"];

	dm->setSignals(signals);
	dm->setLoad(load);

	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), NULL, sigCount, attrSensorSize);
	vector<vector<bool> > results = dm->getLSignals(signals.size());

	request.setMessage("Propagation created");
	request.setData("signals", results);
}

void SimulationHandler::createNpdirs(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	simulation::floyd(dm);
	vector<bool> npdirs = dm->getNPDir();

	request.setMessage("N power dir created");
	request.setData("npdirs", npdirs);
}

void SimulationHandler::createBlocks(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const double delta = request.getDoubleData("delta");
	const vector<vector<int> > dists = request.getInt2dData("dists");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	dm->setDists(dists);
	vector<vector<int> > results = simulation::blocksGPU(dm, delta);
	
	request.setMessage("block GPU created");
	request.setData("blocks", results);
}

void SimulationHandler::createAbduction(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const vector<vector<bool> > signals = request.getBool2dData("signals");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	vector<vector<vector<bool> > > results = simulation::abduction(dm, signals);

	request.setMessage("abduction GPU created");
	request.setData("abduction_even", results[0]);
	request.setData("abduction_odd", results[1]);
}

void SimulationHandler::createPropagateMasks(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();
	simulation::propagateMask(dm);
	vector<vector<bool> > results = dm->getNpdirMasks();

	request.setMessage("Mask propagated");
	request.setData("propagated_mask", results);
}

SimulationHandler::~SimulationHandler() {}