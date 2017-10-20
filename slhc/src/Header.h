#ifndef _HEADER_
#define _HEADER_

#define DATA_LOG_LEVEL logging::DEBUG
#define WORLD_LOG_LEVEL logging::DEBUG
#define MEMORY_EXP 2

/*
The header file to include by almost all files
*/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>

#if defined(_WIN64)
#include<direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

#endif