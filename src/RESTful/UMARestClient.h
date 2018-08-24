#ifndef _UMARESTCLIENT_
#define _UMARESTCLIENT_

#include "RestUtil.h"
#include "cpprest/http_client.h"
#include "cpprest/http_msg.h"
#include "UMARestRequest.h"

using namespace web::http;
using namespace web::http::client;

class DLL_PUBLIC UMARestClient {
private:
	http_client _client;

public:
	UMARestClient(const string &uri);
	~UMARestClient();

	void sendRequest(UMARestRequest &request);
};

#endif