#include "SimulationHandler.h"
#include "logManager.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "UMAException.h"

SimulationHandler::SimulationHandler(string handler_factory, logManager *log_access): AdminHandler(handler_factory, log_access) {
	UMA_DECISION = U("decision");
	UMA_UP = U("up");
	UMA_AMPER = U("amper");
	UMA_DELAY = U("delay");
	UMA_PRUNING = U("pruning");
	UMA_SIGNALS = U("signals");
	UMA_PHI = U("phi");
	UMA_ACTIVE = U("active");
	UMA_AMPER_LIST = U("amper_list");
	UMA_DELAY_LIST = U("delay_list");
	UMA_UUID_LIST = U("uuid_list");
	UMA_IMPLICATION = U("implication");
	UMA_NPDIRS = U("npdirs");
	UMA_SENSORS = U("sensors");

	UMA_OBSPLUS = U("obs_plus");
	UMA_OBSMINUS = U("obs_minus");

	UMA_SAVING = U("saving");
	UMA_LOADING = U("loading");

	UMA_FILE_NAME = U("filename");
	
	UMA_MERGE = U("merge");
}

void SimulationHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/simulation/decision")) {
		create_decision(world, request, response);
		return;
	}
	else if (path == U("/UMA/simulation/amper")) {
		create_amper(world, request, response);
		return;
	}
	else if (path == U("/UMA/simulation/delay")) {
		create_delay(world, request, response);
		return;
	}
	else if (path == U("/UMA/simulation/pruning")) {
		create_pruning(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SimulationHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {

}

void SimulationHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {
}

void SimulationHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {

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


void SimulationHandler::create_decision(World *world, http_request &request, http_response &response) {
	/*
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	double phi = get_double_input(data, UMA_PHI);
	bool active = get_bool_input(data, UMA_ACTIVE);
	vector<bool> obs_plus = get_bool1d_input(data, UMA_OBSPLUS);
	vector<bool> obs_minus = get_bool1d_input(data, UMA_OBSMINUS);

	Agent *agent = world->getAgent(agent_id);
	vector<float> res = agent->decide(obs_plus, obs_minus, phi, active);
	try {
		vector<vector<bool>> current = agent->getCurrent();
		vector<json::value> json_current_plus, json_current_minus;
		vector_bool_to_array(current[0], json_current_plus);
		vector_bool_to_array(current[1], json_current_minus);

		vector<vector<bool>> prediction = agent->getPrediction();
		vector<json::value> json_prediction_plus, json_prediction_minus;
		vector_bool_to_array(prediction[0], json_prediction_plus);
		vector_bool_to_array(prediction[1], json_prediction_minus);

		vector<vector<bool>> target = agent->getTarget();
		vector<json::value> json_target_plus, json_target_minus;
		vector_bool_to_array(target[0], json_target_plus);
		vector_bool_to_array(target[1], json_target_minus);

		json::value return_data;
		json::value message;
		message[MESSAGE] = json::value::string(U("decision made for snapshots"));
		return_data[U("res_plus")] = json::value::number(res[0]);
		return_data[U("res_minus")] = json::value::number(res[1]);
		return_data[U("current_plus")] = json::value::array(json_current_plus);
		return_data[U("current_minus")] = json::value::array(json_current_minus);
		return_data[U("prediction_plus")] = json::value::array(json_prediction_plus);
		return_data[U("prediction_minus")] = json::value::array(json_prediction_minus);
		return_data[U("target_plus")] = json::value::array(json_target_plus);
		return_data[U("target_minus")] = json::value::array(json_target_minus);
		message[U("data")] = return_data;
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		request.reply(status_codes::BadRequest, json::value::string(U("decision made error!")));
	}
	*/
}


void SimulationHandler::create_amper(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<vector<bool>> amper_lists = get_bool2d_input(data, UMA_AMPER_LIST);
	vector<std::pair<string, string> > uuid_lists = get_string_pair1d_input(data, UMA_UUID_LIST);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->ampers(amper_lists, uuid_lists);
	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Amper made succeed"));
	response.set_body(message);
}

void SimulationHandler::create_delay(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<vector<bool>> delay_lists = get_bool2d_input(data, UMA_DELAY_LIST);
	vector<std::pair<string, string> > uuid_lists = get_string_pair1d_input(data, UMA_UUID_LIST);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->delays(delay_lists, uuid_lists);
	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Amper made succeed"));
	response.set_body(message);
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

void SimulationHandler::create_pruning(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<bool> signals = get_bool1d_input(data, UMA_SIGNALS);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->pruning(signals);
	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Pruning made succeed"));
	response.set_body(message);
}

SimulationHandler::~SimulationHandler() {}