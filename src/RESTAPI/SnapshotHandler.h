#ifndef _SNAPSHOTHANDLER_
#define _SNAPSHOTHANDLER_

#include "Global.h"
#include "UMARestHandler.h"
#include "UMARestRequest.h"

class SnapshotHandler : public UMARestHandler {
protected:

public:
	SnapshotHandler(const string &handler_name);
	virtual void handleCreate(UMARestRequest &request);
	virtual void handleUpdate(UMARestRequest &request);
	virtual void handleRead(UMARestRequest &request);
	virtual void handleDelete(UMARestRequest &request);
	void createSnapshot(UMARestRequest &request);
	void createInit(UMARestRequest &request);
	void getSnapshot(UMARestRequest &request);
	void deleteSnapshot(UMARestRequest &request);
	void updateSnapshot(UMARestRequest &request);
	void createAmper(UMARestRequest &request);
	void createDelay(UMARestRequest &request);
	void createPruning(UMARestRequest &request);
	virtual ~SnapshotHandler();
};

#endif