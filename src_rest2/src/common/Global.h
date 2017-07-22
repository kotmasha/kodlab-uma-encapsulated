#ifndef _GLOBAL_
#define _GLOBAL_

#include <string>
#include <cstring>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <exception>
#include <typeinfo>

#if defined(_WIN64)
	#include<direct.h>
#else
	#include <sys/types.h>
	#include <sys/stat.h>
#endif

enum CORE_EXCEPTION {CORE_FATAL, CORE_ERROR, CORE_WARNING};
enum CLIENT_EXCEPTION {CLIENT_FATAL, CLIENT_ERROR, CLIENT_WARNING};

#endif