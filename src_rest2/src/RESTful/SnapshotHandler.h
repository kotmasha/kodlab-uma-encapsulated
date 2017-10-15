#ifndef _SNAPSHOTHANDLER_
#define _SNAPSHOTHANDLER_

#include "Global.h"
#include "AdminHandler.h"

class SnapshotHandler : public AdminHandler {
protected:
	string_t UMA_INITIAL_SIZE;
	string_t UMA_THRESHOLD, UMA_Q, UMA_AUTO_TARGET, UMA_PROPAGATE_MASK;
	string_t UMA_FROM_SENSOR, UMA_TO_SENSOR;
public:
	SnapshotHandler(string handler_factory, logManager *log_access);
	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response);
	void create_snapshot(World *world, http_request &request, http_response &response);
	void create_implication(World *world, http_request &request, http_response &response);
	void get_snapshot(World *world, http_request &request, http_response &response);
	void get_implication(World *world, http_request &request, http_response &response);
	void delete_snapshot(World *world, http_request &request, http_response &response);
	void delete_implication(World *world, http_request &request, http_response &response);
	void update_snapshot(World *world, http_request &request, http_response &response);
	json::value convert_sensor_info(const vector<std::pair<int, pair<string, string> > > &sensor_info);
	json::value convert_size_info(const std::map<string, int> &size_info);
	virtual ~SnapshotHandler();
};

#endif