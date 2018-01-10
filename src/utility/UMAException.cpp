#include "UMAException.h"

//default constructor
UMAException::UMAException(): std::runtime_error("") {}

/*
Another constructor, without status code, if the error is not caused by client rest call
*/
UMAException::UMAException(string message, int error_level, int error_type) : std::runtime_error(message) {
	_error_level = error_level;
	_error_message = message;
	_error_type = error_type;
}

/*
get the error message
output: error message
*/
string UMAException::getErrorMessage() const{
	return _error_message;
}

/*
get the error level
output: error level
*/
int UMAException::getErrorLevel() const{
	return _error_level;
}

/*
get the error type
output: error type
*/
int UMAException::getErrorType() const {
	return _error_type;
}

UMAException::~UMAException() {}