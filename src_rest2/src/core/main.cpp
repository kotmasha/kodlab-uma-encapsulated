#include "Global.h"
#include "listener.h"
#include "World.h"
#include "LogManager.h"
#include "ConfReader.h"

using namespace std;

std::map<string, std::map<string, string>> server_cfg;

int main() {
	LogManager *logManager = new LogManager();
	logManager->init_log_dirs();
	logManager->init_logger();

	server_cfg = ConfReader::read_server_configure();
	string port = server_cfg["Server"]["port"];
	string host = server_cfg["Server"]["host"];

	string url = "http://" + host + ":" + port;

	listener listener(url);
	try
	{
		cout << "launching the server" << endl;
		listener.m_listener.open().wait();
		while (true);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}