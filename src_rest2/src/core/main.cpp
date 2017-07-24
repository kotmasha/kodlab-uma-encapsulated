#include "Global.h"
#include "listener.h"
#include "World.h"

//#include <stdio.h>
using namespace std;

#include<sstream>

string_t convert_string(string s) {
	string_t ss(s.begin(), s.end());
	return ss;
}

void read_ini(string_t &port, string_t &host) {
	try {
		ifstream ini_file("ini.txt");
		string s;
		while (std::getline(ini_file, s)){
			if (s.front() == '#') continue;
			if (s.find("port") != std::string::npos) {
				std::getline(ini_file, s);
				port = convert_string(s);
			}
			else if (s.find("host") != std::string::npos) {
				std::getline(ini_file, s);
				host = convert_string(s);
			}
		}
	}
	catch (exception &e) {
		cout << "Cannot find the ini.txt file, will use default settings" << endl;
	}
}

int main() {
	string_t port = U("8000"), host=U("localhost");
	read_ini(port, host);
	uri url = uri(U("http://") + host + U(":") + port);
	listener listener(url);
	try
	{
		listener.m_listener.open().wait();
		while (true);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}