#ifndef _GLOBAL_
#define _GLOBAL_

//basic include files for the project
#include <string>
#include <cstring>
#include <map>
#include <set>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <exception>
#include <typeinfo>
#include <math.h>
#include <algorithm>
#include <mutex>

using namespace std;

#if defined(_WIN64)
	#include<direct.h>
	#include<Windows.h>
#else
	#include <sys/types.h>
	#include <sys/stat.h>
#endif

#if defined _WIN64 || defined __CYGWIN__
	#ifdef BUILDING_DLL
		#ifdef __GNUC__
			#define DLL_PUBLIC __attribute__ ((dllexport))
		#else
			#define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
		#endif
	#else
		#ifdef __GNUC__
			#define DLL_PUBLIC __attribute__ ((dllimport))
		#else
			#define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
		#endif
	#endif
	#define DLL_LOCAL
#else
	#if __GNUC__ >= 4
		#define DLL_PUBLIC __attribute__ ((visibility ("default")))
		#define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
	#else
		#define DLL_PUBLIC
		#define DLL_LOCAL
	#endif
#endif

#endif //end for _GLOBAL_
