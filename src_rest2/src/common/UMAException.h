#ifndef _UMAEXCEPTION_
#define _UMAEXCEPTION_

#include <string>
#include <stdexcept>
#include "UMAutil.h"

class UMAException: public std::runtime_error {
public:
	enum {ERROR, FATAL};

private:
	int _error_level;
	string _error_message;
	status_code _error_code;

public:
	UMAException(std::string message, int error_level, http::status_code error_code);
	UMAException(std::string message, int error_level);
	UMAException();
	string_t getErrorMessage();
	int getErrorLevel();
	status_code getErrorCode();
};


class CoreException : public UMAException {
public:
	CoreException(std::string message, int error_level, http::status_code error_code);
	CoreException(std::string message, int error_level);
};

class ServerException : public UMAException {
public:
	ServerException(std::string message, int error_level, http::status_code error_code);
	ServerException(std::string message, int error_level);
};

class ClientException : public UMAException {
public:
	ClientException();
	ClientException(std::string message, int error_level, const http::status_code error_code);
};
#endif