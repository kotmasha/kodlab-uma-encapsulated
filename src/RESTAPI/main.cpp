#include "Global.h"
#include "UMARestListener.h"
#include "WorldHandler.h"
#include "ExperimentHandler.h"
#include "AgentHandler.h"
#include "SnapshotHandler.h"
#include "DataHandler.h"
#include "SensorHandler.h"
#include "AttrSensorHandler.h"
#include "SimulationHandler.h"
#include "ConfReader.h"
#include "Logger.h"
#include "PropertyMap.h"

using namespace std;

static WorldHandler *worldHandler = nullptr;
static ExperimentHandler *experimentHandler = nullptr;
static AgentHandler *agentHandler = nullptr;
static SnapshotHandler *snapshotHandler = nullptr;
static DataHandler *dataHandler = nullptr;
static SensorHandler *sensorHandler = nullptr;
static AttrSensorHandler *attrSensorHandler = nullptr;
static SimulationHandler *simulationHandler = nullptr;

static Logger serverLogger("Server", "log/UMA_server.log");

static void initHandlers(UMARestListener &listener) {
	worldHandler = new WorldHandler("world");
	experimentHandler = new ExperimentHandler("experiment");
	agentHandler = new AgentHandler("agent");
	snapshotHandler = new SnapshotHandler("snapshot");
	dataHandler = new DataHandler("data");
	sensorHandler = new SensorHandler("sensor");
	attrSensorHandler = new AttrSensorHandler("attrSensor");
	simulationHandler = new SimulationHandler("simulation");

	listener.registerHandler(worldHandler);
	listener.registerHandler(experimentHandler);
	listener.registerHandler(agentHandler);
	listener.registerHandler(snapshotHandler);
	listener.registerHandler(dataHandler);
	listener.registerHandler(sensorHandler);
	listener.registerHandler(attrSensorHandler);
	listener.registerHandler(simulationHandler);
}


void initHandlerPath(UMARestListener &listener) {
	std::map < string, vector<string>> restMap = ConfReader::readRestmap();
	try {
		for (auto it = restMap.begin(); it != restMap.end(); ++it) {
			string handlerName = it->first;
			for (int i = 0; i < it->second.size(); ++i) {
				string path = it->second[i];
				listener.addPathToHandler(path, handlerName);
			}
		}
	}
	catch (exception &e) {
		serverLogger.error("Having some problem mapping restmap!");
		exit(0);
	}
}


int main() {
	std::map<string, PropertyMap*> serverInfo = ConfReader::readConf("server.ini");
	string port = serverInfo["Server"]->get("port");
	string host = serverInfo["Server"]->get("host");

	string url = "http://" + host + ":" + port;
	serverLogger.info("Will listen on the url " + url);
	UMARestListener listener(url);

	initHandlers(listener);
	initHandlerPath(listener);

	try
	{
		cout << "launching the server" << endl;
		listener.listen();
		while (true);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
