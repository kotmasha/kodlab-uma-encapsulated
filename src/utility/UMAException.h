#ifndef _UMAEXCEPTION_
#define _UMAEXCEPTION_

#include "Global.h"
#include <stdexcept>

using namespace std;

/*
UMA Exception class
*/
class UMAException: public std::runtime_error {
private:
	int _errorLevel;
	int _errorType;
	string _errorMessage;

public:
	enum ERROR_LEVEL {WARN, ERROR, FATAL};
	enum ERROR_TYPE {UNKNOWN, NO_RECORD, DUPLICATE, CONF_ERROR, SERVER, BAD_OPERATION, CLIENT_DATA};

public:
	UMAException();
	UMAException(string message, int errorLevel, int errorType);
	string getErrorMessage() const;
	int getErrorLevel() const;
	int getErrorType() const;
	~UMAException();
};

#endif