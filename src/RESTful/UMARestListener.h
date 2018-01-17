#ifndef _UMA_REST_LISTENER_
#define _UMA_REST_LISTENER_

#include "Global.h"
#include "UMARest.h"

class UMARestHandler;

class DLL_PUBLIC UMARestListener
{
private:
	http_listener _listener;
	std::map<string, UMARestHandler*> _registered_handlers;

public:
	UMARestListener(const string &url);
	~UMARestListener();

	void register_handler(UMARestHandler *handler);
	void add_path_to_handler(const string &path, const string &handler_name);
	void listen();

protected:
	std::map<string, UMARestHandler*> _path_to_handler;

private:
	void handle_get(http_request request);
	void handle_put(http_request request);
	void handle_post(http_request request);
	void handle_delete(http_request request);
	void handle(http_request &request, string request_type);
	UMARestHandler *find_handler(const string &path);
};

#endif
