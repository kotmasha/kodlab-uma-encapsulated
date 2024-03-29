#ifndef _UMA_REST_HANDLER_
#define _UMA_REST_HANDLER_

#include "Global.h"
#include "UMARestRequest.h"

class DLL_PUBLIC UMARestHandler {
protected:
	const string _handlerName;

public:
	UMARestHandler(const string &handlerName);

	virtual void handleCreate(UMARestRequest &request) = 0;
	virtual void handleUpdate(UMARestRequest &request) = 0;
	virtual void handleRead(UMARestRequest &request) = 0;
	virtual void handleDelete(UMARestRequest &request) = 0;

	const string &getHandlerName() const;

	virtual ~UMARestHandler();
};

#endif