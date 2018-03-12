#ifndef _UMARESTTESTFIXTURE_
#define _UMARESTTESTFIXTURE_

#include "gtest/gtest.h"
#include <thread>
#include "cpprest/http_client.h"
#include "cpprest/http_listener.h"
#include "UMARestListener.h"
#include "UMARestHandler.h"
#include "UMARestRequest.h"

using namespace std;
using namespace web;
using namespace web::json;
using namespace web::http;
using namespace web::http::client;

class UMARestTestFixture: public ::testing::Test {
public:
	UMARestTestFixture();
	virtual ~UMARestTestFixture();

	static void serverAction(string actionName);
	static void clientPostAction(string actionName);
	static void clientGetAction(string actionName);
	static void clientReceivingAction(string actionName);
	static void testAction(string actionType);

protected:
	static UMARestListener *listener;
	static UMARestRequest *request;
	static UMARestHandler *handler;
public:
	static json::value clientData;
	static web::uri_builder clientQuery;
	static string clientString;
	static string serverString;
	static int serverInt;
	static int clientInt;
	static double clientDouble;
	static double serverDouble;
	static bool clientBool;
	static bool serverBool;
	static status_code clientStatusCode;
	static status_code serverStatusCode;

	static vector<int> clientInt1d;
	static vector<int> serverInt1d;
	static vector<bool> clientBool1d;
	static vector<bool> serverBool1d;
	static vector<string> clientString1d;
	static vector<string> serverString1d;
	static vector<double> clientDouble1d;
	static vector<double> serverDouble1d;
	static vector<vector<int>> clientInt2d;
	static vector<vector<int>> serverInt2d;
	static vector<vector<bool>> clientBool2d;
	static vector<vector<bool>> serverBool2d;
	static vector<vector<string>> clientString2d;
	static vector<vector<string>> serverString2d;
	static vector<vector<double>> clientDouble2d;
	static vector<vector<double>> serverDouble2d;
	static std::map<string, string> clientMapStringString;
	static std::map<string, string> serverMapStringString;
	static std::map<string, int> clientMapStringInt;
	static std::map<string, int> serverMapStringInt;
	static std::map<string, double> clientMapStringDouble;
	static std::map<string, double> serverMapStringDouble;
	static std::map<string, bool> clientMapStringBool;
	static std::map<string, bool> serverMapStringBool;
};

class UMARestTestHandler: public UMARestHandler {
public:
	UMARestTestHandler();
	~UMARestTestHandler();

	void handle_create(UMARestRequest &request);
	void handle_update(UMARestRequest &request);
	void handle_read(UMARestRequest &request);
	void handle_delete(UMARestRequest &request);
};

#endif