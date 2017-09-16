#include "MatrixHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"

MatrixHandler::MatrixHandler(string handler_factory, logManager *log_access) : AdminHandler(handler_factory, log_access) {
	
}

void MatrixHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {

}

void MatrixHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {

}

void MatrixHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {

}

void MatrixHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/matrix/up")) {
		create_up(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/propagation")) {
		create_propagation(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/npdirs")) {
		create_npdirs(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/ups")) {
		create_ups(world, request, response);
		return;
	}
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void MatrixHandler::create_up(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<bool> signal = get_bool1d_input(data, UMA_SIGNAL);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->up_GPU(signal, false);
	vector<bool> result = snapshot->getUp();
	
	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Up created"));
	message[DATA] = json::value();
	vector<json::value> result_list;
	vector_bool_to_array(result, result_list);
	message[DATA][U("signal")] = json::value::array(result_list);
	response.set_body(message);
}

void MatrixHandler::create_ups(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<vector<bool> > signals = get_bool2d_input(data, UMA_SIGNALS);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	vector<vector<bool> > results = snapshot->ups_GPU(signals);

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Ups created"));
	message[DATA] = json::value();
	vector<json::value> result_lists;
	vector_bool2d_to_array(results, result_lists);
	message[DATA][U("signals")] = json::value::array(result_lists);
	response.set_body(message);
}

void MatrixHandler::create_propagation(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<bool> signal = get_bool1d_input(data, UMA_SIGNAL);
	vector<bool> load = get_bool1d_input(data, UMA_LOAD);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->setLoad(load);
	snapshot->setSignal(signal);
	snapshot->propagate_GPU();
	vector<bool> result = snapshot->getLoad();

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Propagation created"));
	message[DATA] = json::value();
	vector<json::value> result_list;
	vector_bool_to_array(result, result_list);
	message[DATA][U("signal")] = json::value::array(result_list);
	response.set_body(message);
}

void MatrixHandler::create_npdirs(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->floyd_GPU();
	vector<bool> npdirs = snapshot->getNPDir();

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("N power dir created"));
	message[DATA] = json::value();
	vector<json::value> result_list;
	vector_bool_to_array(npdirs, result_list);
	message[DATA][U("npdirs")] = json::value::array(result_list);
	response.set_body(message);
}

MatrixHandler::~MatrixHandler() {}