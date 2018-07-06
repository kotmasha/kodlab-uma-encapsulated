#include "UMAException.h"

//default constructor
UMAException::UMAException(): std::runtime_error("") {}

/*
Another constructor, without status code, if the error is not caused by client rest call
*/
UMAException::UMAException(string message, int errorLevel, int errorType) : std::runtime_error(message) {
	_errorLevel = errorLevel;
	_errorMessage = message;
	_errorType = errorType;
}

/*
get the error message
output: error message
*/
string UMAException::getErrorMessage() const{
	return _errorMessage;
}

/*
get the error level
output: error level
*/
int UMAException::getErrorLevel() const{
	return _errorLevel;
}

/*
get the error type
output: error type
*/
int UMAException::getErrorType() const {
	return _errorType;
}

UMAException::~UMAException() {}