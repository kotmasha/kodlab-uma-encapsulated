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

static void init_handlers(UMARestListener &listener) {
	worldHandler = new WorldHandler("world");
	experimentHandler = new ExperimentHandler("experiment");
	agentHandler = new AgentHandler("agent");
	snapshotHandler = new SnapshotHandler("snapshot");
	dataHandler = new DataHandler("data");
	sensorHandler = new SensorHandler("sensor");
	attrSensorHandler = new AttrSensorHandler("attrSensor");
	simulationHandler = new SimulationHandler("simulation");

	listener.register_handler(worldHandler);
	listener.register_handler(experimentHandler);
	listener.register_handler(agentHandler);
	listener.register_handler(snapshotHandler);
	listener.register_handler(dataHandler);
	listener.register_handler(sensorHandler);
	listener.register_handler(attrSensorHandler);
	listener.register_handler(simulationHandler);
}


void init_handler_path(UMARestListener &listener) {
	std::map < string, vector<string>> rest_map = ConfReader::readRestmap();
	try {
		for (auto it = rest_map.begin(); it != rest_map.end(); ++it) {
			string handler_name = it->first;
			for (int i = 0; i < it->second.size(); ++i) {
				string path = it->second[i];
				listener.add_path_to_handler(path, handler_name);
			}
		}
	}
	catch (exception &e) {
		serverLogger.error("Having some problem mapping restmap!");
		exit(0);
	}
}


int main() {
	std::map<string, std::map<string, string>> server_info = ConfReader::readConf("server.ini");
	string port = server_info["Server"]["port"];
	string host = server_info["Server"]["host"];

	string url = "http://" + host + ":" + port;
	serverLogger.info("Will listen on the url " + url);
	UMARestListener listener(url);

	init_handlers(listener);
	init_handler_path(listener);

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
