#include "Global.h"
#include "listener.h"
#include "World.h"

int main() {
	listener listener(L"http://localhost:8000");
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