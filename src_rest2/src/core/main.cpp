#include "Global.h"
#include "listener.h"
#include "World.h"

//#include <stdio.h>
using namespace std;

int main() {
	string_t url = U("http://localhost:8000");
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
	system("pause");
}