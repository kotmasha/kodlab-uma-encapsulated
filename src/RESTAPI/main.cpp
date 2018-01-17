#include "Global.h"
#include "UMARestListener.h"
#include "WorldHandler.h"
#include "AgentHandler.h"
#include "SnapshotHandler.h"
#include "DataHandler.h"
#include "SensorHandler.h"
#include "MeasurableHandler.h"
#include "SimulationHandler.h"
#include "ConfReader.h"
#include "Logger.h"

using namespace std;

static WorldHandler *world_handler = NULL;
static AgentHandler *agent_handler = NULL;
static SnapshotHandler *snapshot_handler = NULL;
static DataHandler *data_handler = NULL;
static SensorHandler *sensor_handler = NULL;
static MeasurableHandler *measurable_handler = NULL;
static SimulationHandler *simulation_handler = NULL;

static Logger serverLogger("Server", "log/UMA_server.log");

static void init_handlers(UMARestListener &listener) {
	world_handler = new WorldHandler("world");
	agent_handler = new AgentHandler("agent");
	snapshot_handler = new SnapshotHandler("snapshot");
	data_handler = new DataHandler("data");
	sensor_handler = new SensorHandler("sensor");
	measurable_handler = new MeasurableHandler("measurable");
	simulation_handler = new SimulationHandler("simulation");

	listener.register_handler(world_handler);
	listener.register_handler(agent_handler);
	listener.register_handler(snapshot_handler);
	listener.register_handler(data_handler);
	listener.register_handler(sensor_handler);
	listener.register_handler(measurable_handler);
	listener.register_handler(simulation_handler);
}


void init_handler_path(UMARestListener &listener) {
	std::map < string, vector<string>> rest_map = ConfReader::read_restmap();
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
	std::map<string, std::map<string, string>> server_info = ConfReader::read_conf("server.ini");
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