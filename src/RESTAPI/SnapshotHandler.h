#ifndef _SNAPSHOTHANDLER_
#define _SNAPSHOTHANDLER_

#include "Global.h"
#include "UMARestHandler.h"
#include "UMARestRequest.h"

class SnapshotHandler : public UMARestHandler {
protected:

public:
	SnapshotHandler(const string &handler_name);
	virtual void handle_create(UMARestRequest &request);
	virtual void handle_update(UMARestRequest &request);
	virtual void handle_read(UMARestRequest &request);
	virtual void handle_delete(UMARestRequest &request);
	void create_snapshot(UMARestRequest &request);
	void create_init(UMARestRequest &request);
	void get_snapshot(UMARestRequest &request);
	void delete_snapshot(UMARestRequest &request);
	void update_snapshot(UMARestRequest &request);
	void create_amper(UMARestRequest &request);
	void create_delay(UMARestRequest &request);
	void create_pruning(UMARestRequest &request);
	virtual ~SnapshotHandler();
};

#endif