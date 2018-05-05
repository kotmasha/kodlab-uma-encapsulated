#include "UMARestClient.h"

UMARestClient::UMARestClient(const string &uri): _client(web::uri(RestUtil::string_to_string_t(uri))) {
}

void UMARestClient::send_request(UMARestRequest &request){
	_client.request(request._request).then([&](http_response response) {
		request._response = response;
	}).wait();
}

UMARestClient::~UMARestClient() {}
