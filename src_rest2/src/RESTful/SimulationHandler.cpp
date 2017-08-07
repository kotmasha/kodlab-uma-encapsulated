#include "SimulationHandler.h"
#include "logManager.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"

SimulationHandler::SimulationHandler(logManager *log_access): AdminHandler(log_access) {
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

void SimulationHandler::handle_create(World *world, vector<string_t> &paths, http_request &request) {
	json::value data = request.extract_json().get();
	if (paths[0] == UMA_DECISION) {
		create_decision(world, data, request);
		return;
	}
	else if (paths[0] == UMA_DELAY) {
		create_delay(world, data, request);
		return;
	}
	else if (paths[0] == UMA_AMPER) {
		create_amper(world, data, request);
		return;
	}
	else if (paths[0] == UMA_UP) {
		create_up(world, data, request);
		return;
	}
	else if (paths[0] == UMA_SAVING) {
		create_saving(world, data, request);
		return;
	}
	else if (paths[0] == UMA_LOADING) {
		create_loading(world, data, request);
		return;
	}
	else if (paths[0] == UMA_PRUNING) {
		create_pruning(world, data, request);
		return;
	}
	else if (paths[0] == UMA_MERGE) {
		create_merging(world, data, request);
	}
	else if (paths[0] == UMA_NPDIRS) {
		create_npdirs(world, data, request);
	}
	else if (paths[0] == UMA_IMPLICATION) {
		create_implication(world, data, request);
	}

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
	json::value message;
	message[MESSAGE] = json::value::string(U("cannot handle ") + paths[0] + U(" object"));
	request.reply(status_codes::BadRequest, message);
}

void SimulationHandler::handle_update(World *world, vector<string_t> &paths, http_request &request) {

}

void SimulationHandler::handle_read(World *world, vector<string_t> &paths, http_request &request) {
}

void SimulationHandler::handle_delete(World *world, vector<string_t> &paths, http_request &request) {

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


void SimulationHandler::create_decision(World *world, json::value &data, http_request &request) {
	string agent_id;
	double phi;
	bool active;
	vector<bool> obs_plus;
	vector<bool> obs_minus;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		phi = get_double_input(data, UMA_PHI, request);
		active = get_bool_input(data, UMA_ACTIVE, request);
		obs_plus = get_bool1d_input(data, UMA_OBSPLUS, request);
		obs_minus = get_bool1d_input(data, UMA_OBSMINUS, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	Agent *agent = NULL;
	Snapshot *snapshot = NULL;
	if (!get_agent_by_id(world, agent_id, agent, request)) return;
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
}

void SimulationHandler::create_amper(World *world, json::value &data, http_request &request) {
	string agent_id, snapshot_id;
	vector<vector<bool>> amper_lists;
	vector<std::pair<string, string> > uuid_lists;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
		amper_lists = get_bool2d_input(data, UMA_AMPER_LIST, request);
		uuid_lists = get_string_pair1d_input(data, UMA_UUID_LIST, request);
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
		snapshot->ampers(amper_lists, uuid_lists);
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		json::value message;
		message[MESSAGE] = json::value::string(U("amper made successful"));
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[MESSAGE] = json::value::string(U("amper made error!"));
		request.reply(status_codes::BadRequest, message);
	}
}

void SimulationHandler::create_delay(World *world, json::value &data, http_request &request) {
	string agent_id, snapshot_id;
	vector<vector<bool>> delay_lists;
	vector<std::pair<string, string> > uuid_lists;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
		delay_lists = get_bool2d_input(data, UMA_DELAY_LIST, request);
		uuid_lists = get_string_pair1d_input(data, UMA_UUID_LIST, request);
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
		snapshot->delays(delay_lists, uuid_lists);
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		json::value message;
		message[MESSAGE] = json::value::string(U("delay made successful"));
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[MESSAGE] = json::value::string(U("delay made error"));
		request.reply(status_codes::BadRequest, message);
	}
}

void SimulationHandler::create_up(World *world, json::value &data, http_request &request) {
	string agent_id, snapshot_id;
	vector<bool> signals;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
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
		snapshot->up_GPU(signals, false);
		vector<bool> up = snapshot->getUp();
		vector<json::value> json_up;
		vector_bool_to_array(up, json_up);
		json::value return_data = json::value::array(json_up);
		json::value message;

		message[MESSAGE] = json::value::string(U("up made successful"));
		message[U("data")] = return_data;
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		request.reply(status_codes::Created, message);
		return;
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[MESSAGE] = json::value::string(U("up made error!"));
		request.reply(status_codes::BadRequest, message);
	}
}

void SimulationHandler::create_npdirs(World *world, json::value &data, http_request &request) {
	string agent_id, snapshot_id;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
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
		snapshot->floyd_GPU();
		vector<int> npdirs = snapshot->getNPDir();
		vector<json::value> json_npdirs;
		vector_int_to_array(npdirs, json_npdirs);
		json::value return_data = json::value::array(json_npdirs);
		json::value message;

		message[MESSAGE] = json::value::string(U("npdirs made successful"));
		message[U("data")] = return_data;
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		request.reply(status_codes::Created, message);
		return;
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[MESSAGE] = json::value::string(U("npdirs made error!"));
		request.reply(status_codes::BadRequest, message);
	}
}

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

void SimulationHandler::create_pruning(World *world, json::value &data, http_request &request) {
	string agent_id, snapshot_id;
	vector<bool> signals;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
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
		snapshot->pruning(signals);
		json::value message;

		message[MESSAGE] = json::value::string(U("pruning made successful"));
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		request.reply(status_codes::Created, message);
		return;
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[MESSAGE] = json::value::string(U("pruning made error!"));
		request.reply(status_codes::BadRequest, message);
	}
}

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

SimulationHandler::~SimulationHandler() {}