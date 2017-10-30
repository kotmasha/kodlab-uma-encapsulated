#include "MatrixHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "Simulation.h"

MatrixHandler::MatrixHandler(string handler_factory, logManager *log_access) : AdminHandler(handler_factory, log_access) {
	UMA_BLOCK_DISTS = U("dists");
	UMA_BLOCK_DT = U("delta");
}

void MatrixHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {

}

void MatrixHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {

}

void MatrixHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {

}

void MatrixHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/matrix/propagation")) {
		create_propagation(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/npdirs")) {
		create_npdirs(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/up")) {
		create_up(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/ups")) {
		create_ups(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/blocks")) {
		create_blocks(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/abduction")) {
		create_abduction(world, request, response);
		return;
	}
	else if (path == U("/UMA/matrix/propagate_masks")) {
		create_propagated_masks(world, request, response);
		return;
	}
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}


void MatrixHandler::create_up(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<bool> signal = get_bool1d_input(data, UMA_SIGNAL);
	vector<vector<bool> > signals(1, signal);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	int measurable_size = dm->getMeasurableSize();
	dm->setSignals(signals);
	simulation::ups_GPU(dm->_dvar_b(NPDIRS), dm->_dvar_b(SIGNALS), NULL, 1, measurable_size);

	vector<vector<bool> > results = dm->getSignals(1);

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Up created"));
	message[DATA] = json::value();
	vector<json::value> result_list;
	vector_bool_to_array(results[0], result_list);
	message[DATA][U("signal")] = json::value::array(result_list);
	response.set_body(message);
}

void MatrixHandler::create_ups(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<vector<bool> > signals = get_bool2d_input(data, UMA_SIGNALS);
	int sig_count = signals.size();

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	int measurable_size = dm->getMeasurableSize();
	dm->setSignals(signals);
	simulation::ups_GPU(dm->_dvar_b(NPDIRS), dm->_dvar_b(SIGNALS), NULL, sig_count, measurable_size);

	vector<vector<bool> > results = dm->getSignals(signals.size());

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
	vector<vector<bool> > signals = get_bool2d_input(data, UMA_SIGNALS);
	vector<bool> load = get_bool1d_input(data, UMA_LOAD);
	int sig_count = signals.size();

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	int measurable_size = dm->getMeasurableSize();
	dm->setSignals(signals);
	dm->setLoad(load);
	simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(SIGNALS), dm->_dvar_b(LSIGNALS), NULL, sig_count, measurable_size);
	vector<vector<bool> > results = dm->getLSignals(signals.size());

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Propagation created"));
	message[DATA] = json::value();
	vector<json::value> result_lists;
	vector_bool2d_to_array(results, result_lists);
	message[DATA][U("signals")] = json::value::array(result_lists);
	response.set_body(message);
}

void MatrixHandler::create_npdirs(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	simulation::floyd(dm);
	vector<bool> npdirs = dm->getNPDir();

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("N power dir created"));
	message[DATA] = json::value();
	vector<json::value> result_list;
	vector_bool_to_array(npdirs, result_list);
	message[DATA][U("npdirs")] = json::value::array(result_list);
	response.set_body(message);
}

void MatrixHandler::create_blocks(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	double delta = get_double_input(data, UMA_BLOCK_DT);
	vector<vector<int> > dists = get_int2d_input(data, UMA_BLOCK_DISTS);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	dm->setDists(dists);
	vector<vector<int> > results = simulation::blocks_GPU(dm, delta);

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("block GPU created"));
	message[DATA] = json::value();
	vector<json::value> result_list;
	vector_int2d_to_array(results, result_list);
	message[DATA][U("blocks")] = json::value::array(result_list);
	response.set_body(message);
}

void MatrixHandler::create_abduction(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	vector<vector<bool> > signals = get_bool2d_input(data, UMA_SIGNALS);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	vector<vector<vector<bool> > > results = simulation::abduction(dm, signals);

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("abduction GPU created"));
	message[DATA] = json::value();
	vector<json::value> result_list_even, result_list_odd;

	vector_bool2d_to_array(results[0], result_list_even);
	message[DATA][U("abduction_even")] = json::value::array(result_list_even);

	vector_bool2d_to_array(results[1], result_list_odd);
	message[DATA][U("abduction_odd")] = json::value::array(result_list_odd);
	response.set_body(message);
}

void MatrixHandler::create_propagated_masks(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();
	simulation::propagate_mask(dm);
	vector<vector<bool> > results = dm->getNpdirMasks();

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("mask propagated"));
	message[DATA] = json::value();
	vector<json::value> result_list;
	vector_bool2d_to_array(results, result_list);
	message[DATA][U("propagated_mask")] = json::value::array(result_list);
	response.set_body(message);
}


MatrixHandler::~MatrixHandler() {}