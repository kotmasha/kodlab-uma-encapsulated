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
#include "PropertyPage.h"

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
	PropertyPage *restmap = ConfReader::readRestmap();
	try {
		for (auto it = restmap->begin(); it != restmap->end(); ++it) {
			string handlerName = it->first;
			PropertyMap *pm = it->second;
			for (auto pmIt = pm->begin(); pmIt != pm->end(); ++pmIt) {
				string path = pmIt->first;
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
	PropertyPage *serverInfo = ConfReader::readConf("server.ini");
	string port = serverInfo->get("Server")->get("port");
	string host = serverInfo->get("Server")->get("host");

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
