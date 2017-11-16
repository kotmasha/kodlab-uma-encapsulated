#ifndef _UMAEXCEPTION_
#define _UMAEXCEPTION_

#include <string>
#include <stdexcept>
#include "UMAutil.h"

using namespace std;

class UMAException: public std::runtime_error {
private:
	int _error_level;
	string _error_message;
	status_code _error_code;

public:
	UMAException(std::string message, int error_level, http::status_code error_code);
	UMAException(std::string message, int error_level);
	UMAException();
	string getErrorMessage();
	int getErrorLevel();
	status_code getErrorCode();
};


class CoreException : public UMAException {
public:
	enum { CORE_ERROR, CORE_FATAL };
	CoreException(std::string message, int error_level, http::status_code error_code);
	CoreException(std::string message, int error_level);
};

class ServerException : public UMAException {
public:
	enum { SERVER_ERROR, SERVER_FATAL };
	ServerException(std::string message, int error_level, http::status_code error_code);
	ServerException(std::string message, int error_level);
};

class ClientException : public UMAException {
public:
	enum { CLIENT_ERROR };
	ClientException();
	ClientException(std::string message, int error_level, const http::status_code error_code);
};
#endif