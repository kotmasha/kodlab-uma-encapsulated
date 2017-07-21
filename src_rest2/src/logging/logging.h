#ifndef _LOGGING_
#define _LOGGING_

#include "Global.h"
using namespace std;

class logManager;

class logging{
protected:
	string _filename;
	string _classname;
	bool _active;
	string _level;
	ofstream *_output;
public:
	logging();
	logging(const logging &l);
	logging(string filename, string classname);
	string parse_class_name(string classname);
	void operator<<(string info);
	void operator<<(wstring info);
	enum LOG_LEVEL{ERROR, WARN, INFO, DEBUG, VERBOSE};
	friend class logManager;
};

#endif