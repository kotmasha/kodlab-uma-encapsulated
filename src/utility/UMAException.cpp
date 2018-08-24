#include "UMAException.h"

/*#######################UMAException#########################*/

//default constructor
UMAException::UMAException(): std::runtime_error(""), _type(UMA_UNKNOWN) {
	_isFatal = false;
	_log = nullptr;
}

/*
Another constructor, with error msg only
*/
UMAException::UMAException(string message, UMAExceptionType type) : std::runtime_error(message), _type(type) {
	_errorMessage = message;
	_isFatal = false;
	_log = nullptr;
}

/*
Another constructor, with error msg and isFatal bool
*/
UMAException::UMAException(string message, bool isFatal, UMAExceptionType type) :
	std::runtime_error(message), _type(type) {
	_errorMessage = message;
	_isFatal = isFatal;
	_log = nullptr;
}

/*
Another constructor, with error msg, isFatal bool and log pointer
*/
UMAException::UMAException(string message, bool isFatal, Logger *log, UMAExceptionType type) :
	std::runtime_error(message), _type(type) {
	_errorMessage = message;
	_isFatal = isFatal;
	_log = log;

	if (log) {
		log->error(message);
	}
}

/*
get the error message
output: error message
*/
string UMAException::getErrorMessage() const{
	return _errorMessage;
}

/*
return the type of UMAException
*/
const UMAException::UMAExceptionType UMAException::getType() const {
	return _type;
}

/*
function to get the isFatal variable
*/
const bool UMAException::isFatal() const {
	return _isFatal;
}

/*
function to check whether the error msg is logged, basically check _log variable
*/
bool UMAException::isErrorLogged() {
	return !(nullptr == _log);
}

UMAException::~UMAException() {}

/*#######################UMAException#########################*/

/*#######################OTHER Exception#########################*/

UMAInternalException::UMAInternalException(string message, bool isFatal, Logger *log) :
	UMAException(message, isFatal, nullptr, UMA_INTERNAL) {}

UMAInternalException::~UMAInternalException() {}

string UMAInternalException::getErrorMessage() const {
	return "Internal Error Caught: " + UMAException::getErrorMessage();
}


UMAInvalidArgsException::UMAInvalidArgsException(string message, bool isFatal, Logger *log) :
	UMAException(message, isFatal, nullptr, UMA_INVALID_ARGS) {}

UMAInvalidArgsException::~UMAInvalidArgsException() {}

string UMAInvalidArgsException::getErrorMessage() const {
	return "Invalid Argument Error Caught: " + UMAException::getErrorMessage();
}


UMANoResourceException::UMANoResourceException(string message, bool isFatal, Logger *log) :
	UMAException(message, isFatal, nullptr, UMA_NO_RESOURCE) {}

UMANoResourceException::~UMANoResourceException() {}

string UMANoResourceException::getErrorMessage() const {
	return "No Resource Error Caught: " + UMAException::getErrorMessage();
}

UMADuplicationException::UMADuplicationException(string message, bool isFatal, Logger *log) :
	UMAException(message, isFatal, nullptr, UMA_DUPLICATION) {}

UMADuplicationException::~UMADuplicationException() {}

string UMADuplicationException::getErrorMessage() const {
	return "Duplication Record Error Caught: " + UMAException::getErrorMessage();
}

UMABadOperationException::UMABadOperationException(string message, bool isFatal, Logger *log) :
	UMAException(message, isFatal, nullptr, UMA_BAD_OPERATION) {}

UMABadOperationException::~UMABadOperationException() {}

string UMABadOperationException::getErrorMessage() const {
	return "Duplication Record Error Caught: " + UMAException::getErrorMessage();
}

/*#######################OTHER Exception#########################*/