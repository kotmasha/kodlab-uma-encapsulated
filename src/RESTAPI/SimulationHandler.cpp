#include "SimulationHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "Simulation.h"
#include "UMAException.h"

SimulationHandler::SimulationHandler(const string &handler_name): UMARestHandler(handler_name) {
}

void SimulationHandler::handle_create(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/simulation/decision") {
		create_decision(request);
		return;
	}
	else if (request_url == "/UMA/simulation/up") {
		create_up(request);
		return;
	}
	else if (request_url == "/UMA/simulation/ups") {
		create_ups(request);
		return;
	}
	else if (request_url == "/UMA/simulation/propagation") {
		create_propagation(request);
		return;
	}
	else if (request_url == "/UMA/simulation/npdirs") {
		create_npdirs(request);
		return;
	}
	else if (request_url == "/UMA/simulation/downs") {
		create_downs(request);
		return;
	}
	else if (request_url == "/UMA/simulation/blocks") {
		create_blocks(request);
		return;
	}
	else if (request_url == "/UMA/simulation/abduction") {
		create_abduction(request);
		return;
	}
	else if (request_url == "/UMA/simulation/propagate_masks") {
		create_propagate_masks(request);
		return;
	}

	throw UMAException("Cannot handle POST " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SimulationHandler::handle_update(UMARestRequest &request) {
	throw UMAException("Cannot handle PUT " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SimulationHandler::handle_read(UMARestRequest &request) {
	throw UMAException("Cannot handle GET " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SimulationHandler::handle_delete(UMARestRequest &request) {
	throw UMAException("Cannot handle DELETE " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

/*
//this version is on snapshot layer
void SimulationHandler::create_decision(World *world, json::value &data, http_request &request) {
	string agent_id, snapshot_id;
	double phi;
	bool active;
	vector<bool> signals;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
		phi = get_double_input(data, UMA_PHI, request);
		active = get_bool_input(data, UMA_ACTIVE, request);
		signals = get_bool1d_input(data, UMA_SIGNALS, request);
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
		float res = snapshot->decide(signals, phi, active);
		vector<bool> current = snapshot->getCurrent();
		vector<json::value> json_current;
		vector_bool_to_array(current, json_current);

		vector<bool> prediction = snapshot->getPrediction();
		vector<json::value> json_prediction;
		vector_bool_to_array(prediction, json_prediction);

		vector<bool> target = snapshot->getTarget();
		vector<json::value> json_target;
		vector_bool_to_array(target, json_target);

		json::value return_data;
		json::value message;
		message[MESSAGE] = json::value::string(U("decision made"));
		return_data[U("res")] = json::value::number(res);
		return_data[U("current")] = json::value::array(json_current);
		return_data[U("prediction")] = json::value::array(json_prediction);
		return_data[U("target")] = json::value::array(json_target);
		message[U("data")] = return_data;
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400"0;
		request.reply(status_codes::BadRequest, json::value::string(U("decision made error!")));
	}
}

*/


void SimulationHandler::create_decision(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const double phi = request.get_double_data("phi");
	const bool active = request.get_bool_data("active");
	const vector<bool> obs_plus = request.get_bool1d_data("obs_plus");
	const vector<bool> obs_minus = request.get_bool1d_data("obs_minus");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot_plus = agent->getSnapshot("plus");
	Snapshot *snapshot_minus = agent->getSnapshot("minus");
	vector<float> res = simulation::decide(agent, obs_plus, obs_minus, phi, active);

	vector<vector<bool>> current, prediction, target;
	current.push_back(snapshot_plus->getDM()->getCurrent());
	current.push_back(snapshot_minus->getDM()->getCurrent());
	prediction.push_back(snapshot_plus->getDM()->getPrediction());
	prediction.push_back(snapshot_minus->getDM()->getPrediction());
	target.push_back(snapshot_plus->getDM()->getTarget());
	target.push_back(snapshot_minus->getDM()->getTarget());

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


void SimulationHandler::create_up(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");
	const vector<bool> signal = request.get_bool1d_data("signal");
	vector<vector<bool> > signals(1, signal);

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attr_sensor_size"];
	dm->setSignals(signals);
	simulation::ups_GPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, 1, attr_sensor_size);

	vector<vector<bool> > results = dm->getSignals(1);

	request.set_message("Up created");
	request.set_data("signal", results[0]);
}

void SimulationHandler::create_ups(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");
	const vector<vector<bool> > signals = request.get_bool2d_data("signals");
	int sig_count = signals.size();

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attr_sensor_size"];

	dm->setSignals(signals);
	simulation::ups_GPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, sig_count, attr_sensor_size);

	vector<vector<bool> > results = dm->getSignals(signals.size());

	request.set_message("Ups created");
	request.set_data("signals", results);
}

void SimulationHandler::create_downs(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");
	const vector<vector<bool> > signals = request.get_bool2d_data("signals");
	int sig_count = signals.size();

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attr_sensor_size"];

	dm->setSignals(signals);
	simulation::downs_GPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, sig_count, attr_sensor_size);

	vector<vector<bool> > results = dm->getSignals(signals.size());

	request.set_message("Downs created");
	request.set_data("signals", results);
}

void SimulationHandler::create_propagation(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");
	const vector<vector<bool> > signals = request.get_bool2d_data("signals");
	const vector<bool> load = request.get_bool1d_data("load");
	int sig_count = signals.size();

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attr_sensor_size"];

	dm->setSignals(signals);
	dm->setLoad(load);

	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), NULL, sig_count, attr_sensor_size);
	vector<vector<bool> > results = dm->getLSignals(signals.size());

	request.set_message("Propagation created");
	request.set_data("signals", results);
}

void SimulationHandler::create_npdirs(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	simulation::floyd(dm);
	vector<bool> npdirs = dm->getNPDir();

	request.set_message("N power dir created");
	request.set_data("npdirs", npdirs);
}

void SimulationHandler::create_blocks(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");
	const double delta = request.get_double_data("delta");
	const vector<vector<int> > dists = request.get_int2d_data("dists");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	dm->setDists(dists);
	vector<vector<int> > results = simulation::blocks_GPU(dm, delta);
	
	request.set_message("block GPU created");
	request.set_data("blocks", results);
}

void SimulationHandler::create_abduction(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");
	const vector<vector<bool> > signals = request.get_bool2d_data("signals");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	vector<vector<vector<bool> > > results = simulation::abduction(dm, signals);

	request.set_message("abduction GPU created");
	request.set_data("abduction_even", results[0]);
	request.set_data("abduction_odd", results[1]);
}

void SimulationHandler::create_propagate_masks(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	simulation::propagate_mask(dm);
	vector<vector<bool> > results = dm->getNpdirMasks();

	request.set_message("Mask propagated");
	request.set_data("propagated_mask", results);
}

SimulationHandler::~SimulationHandler() {}