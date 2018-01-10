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
	int _error_level;
	int _error_type;
	string _error_message;

public:
	enum ERROR_LEVEL {WARN, ERROR, FATAL};
	enum ERROR_TYPE {UNKNOWN, NO_RECORD, DUPLICATE, CONF_ERROR, SERVER, BAD_OPERATION, CLIENT_DATA};

public:
	UMAException();
	UMAException(string message, int error_level, int error_type);
	string getErrorMessage() const;
	int getErrorLevel() const;
	int getErrorType() const;
	~UMAException();
};

#endif