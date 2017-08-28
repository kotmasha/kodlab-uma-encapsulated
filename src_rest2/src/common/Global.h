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

#endif