#ifndef _UMA_REST_LISTENER_
#define _UMA_REST_LISTENER_

#include "Global.h"
#include "UMARest.h"

class UMARestHandler;

class DLL_PUBLIC UMARestListener
{
private:
	http_listener _listener;
	std::map<string, UMARestHandler*> _registeredHandlers;

public:
	UMARestListener(const string &url);
	~UMARestListener();

	void registerHandler(UMARestHandler *handler);
	void addPathToHandler(const string &path, const string &handlerName);
	void listen();

protected:
	std::map<string, UMARestHandler*> _pathToHandler;

private:
	void handleGet(http_request request);
	void handlePut(http_request request);
	void handlePost(http_request request);
	void handleDelete(http_request request);
	void handle(http_request &request, string requestType);
	UMARestHandler *findHandler(const string &path);
};

#endif
