#include "SimulationHandler.h"
#include "logManager.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"

SimulationHandler::SimulationHandler(logManager *log_access): AdminHandler(log_access) {
	UMA_DECISION = L"decision";
	UMA_UP = L"up";
	UMA_AMPER = L"amper";
	UMA_DELAY = L"delay";
	UMA_PRUNING = L"pruning";
	UMA_SIGNALS = L"signals";
	UMA_PHI = L"phi";
	UMA_ACTIVE = L"active";
	UMA_AMPER_LIST = L"amper_list";
	UMA_DELAY_LIST = L"delay_list";
	UMA_UUID_LIST = L"uuid_list";

	UMA_SAVING = L"saving";
	UMA_LOADING = L"loading";

	UMA_FILE_NAME = L"filename";
	
	UMA_MERGE = L"merge";
}

void SimulationHandler::handle_create(World *world, vector<string_t> &paths, http_request &request) {
	json::value &data = request.extract_json().get();
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

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
	json::value message;
	message[MESSAGE] = json::value::string(L"cannot handle " + paths[0] + L" object");
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
		message[MESSAGE] = json::value::string(L"decision made");
		return_data[L"res"] = json::value::number(res);
		return_data[L"current"] = json::value::array(json_current);
		return_data[L"prediction"] = json::value::array(json_prediction);
		return_data[L"target"] = json::value::array(json_target);
		message[L"data"] = return_data;
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		request.reply(status_codes::BadRequest, json::value::string(L"decision made error!"));
	}
}
*/


void SimulationHandler::create_decision(World *world, json::value &data, http_request &request) {
	string agent_id;
	double phi;
	bool active;
	vector<bool> signals;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
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
	vector<float> res = agent->decide(signals, phi, active);
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
		message[MESSAGE] = json::value::string(L"decision made for snapshots");
		return_data[L"res_plus"] = json::value::number(res[0]);
		return_data[L"res_minus"] = json::value::number(res[1]);
		return_data[L"current_plus"] = json::value::array(json_current_plus);
		return_data[L"current_minus"] = json::value::array(json_current_minus);
		return_data[L"prediction_plus"] = json::value::array(json_prediction_plus);
		return_data[L"prediction_minus"] = json::value::array(json_prediction_minus);
		return_data[L"target_plus"] = json::value::array(json_target_plus);
		return_data[L"target_minus"] = json::value::array(json_target_minus);
		message[L"data"] = return_data;
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		request.reply(status_codes::BadRequest, json::value::string(L"decision made error!"));
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
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		json::value message;
		message[MESSAGE] = json::value::string(L"amper made successful");
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"amper made error!");
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
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		json::value message;
		message[MESSAGE] = json::value::string(L"delay made successful");
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"delay made error");
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

		message[MESSAGE] = json::value::string(L"up made successful");
		message[L"data"] = return_data;
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, message);
		return;
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"up made error!");
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
		message[MESSAGE] = json::value::string(L"data saved successfully to " + data[UMA_FILE_NAME].as_string() + L".uma");
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"cannot saving data!");
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
		message[MESSAGE] = json::value::string(L"data loaded successfully loaded from " + data[UMA_FILE_NAME].as_string() + L".uma");
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, message);
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"cannot load data!");
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

		message[MESSAGE] = json::value::string(L"pruning made successful");
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, message);
		return;
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"pruning made error!");
		request.reply(status_codes::BadRequest, message);
	}
}

void SimulationHandler::create_merging(World *world, json::value &data, http_request &request) {
	try {
		world->merge_test();
		json::value message;

		message[MESSAGE] = json::value::string(L"test data merged successfully");
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, message);
		return;
	}
	catch (exception &e) {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"test data merged error!");
		request.reply(status_codes::BadRequest, message);
	}
}

SimulationHandler::~SimulationHandler() {}