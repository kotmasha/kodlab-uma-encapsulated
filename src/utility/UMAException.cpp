#include "UMAException.h"

/*#######################UMAException#########################*/

//default constructor
UMAException::UMAException(): std::runtime_error("") {
	_isFatal = false;
}

/*
Another constructor, with error msg only
*/
UMAException::UMAException(string message) : std::runtime_error(message) {
	_errorMessage = message;
	_isFatal = false;
	_log = nullptr;
}

/*
Another constructor, with error msg and isFatal bool
*/
UMAException::UMAException(string message, bool isFatal) : std::runtime_error(message) {
	_errorMessage = message;
	_isFatal = isFatal;
	_log = nullptr;
}

/*
Another constructor, with error msg, isFatal bool and log pointer
*/
UMAException::UMAException(string message, bool isFatal, Logger *log) : std::runtime_error(message) {
	_errorMessage = message;
	_isFatal = isFatal;
	_log = log;
}

/*
get the error message
output: error message
*/
string UMAException::getErrorMessage() const{
	return _errorMessage;
}

/*
function to get the isFatal variable
*/
const bool UMAException::isFatal() const {
	return _isFatal;
}

UMAException::~UMAException() {}

/*#######################UMAException#########################*/

/*#######################OTHER Exception#########################*/

UMAInternalException::UMAInternalException(string message, bool isFatal, Logger *log) : UMAException(message, isFatal, nullptr) {}

UMAInternalException::~UMAInternalException() {}

string UMAInternalException::getErrorMessage() const {
	return "Internal Error Caught: " + UMAException::getErrorMessage();
}


UMAInvalidArgsException::UMAInvalidArgsException(string message, bool isFatal, Logger *log) : UMAException(message, isFatal, nullptr) {}

UMAInvalidArgsException::~UMAInvalidArgsException() {}

string UMAInvalidArgsException::getErrorMessage() const {
	return "Invalid Argument Error Caught: " + UMAException::getErrorMessage();
}


UMANoResourceException::UMANoResourceException(string message, bool isFatal, Logger *log) : UMAException(message, isFatal, nullptr) {}

UMANoResourceException::~UMANoResourceException() {}

string UMANoResourceException::getErrorMessage() const {
	return "No Resource Error Caught: " + UMAException::getErrorMessage();
}

UMADuplicationException::UMADuplicationException(string message, bool isFatal, Logger *log) : UMAException(message, isFatal, nullptr) {}

UMADuplicationException::~UMADuplicationException() {}

string UMADuplicationException::getErrorMessage() const {
	return "Duplication Record Error Caught: " + UMAException::getErrorMessage();
}

UMABadOperationException::UMABadOperationException(string message, bool isFatal, Logger *log) : UMAException(message, isFatal, nullptr) {}

UMABadOperationException::~UMABadOperationException() {}

string UMABadOperationException::getErrorMessage() const {
	return "Duplication Record Error Caught: " + UMAException::getErrorMessage();
}

/*#######################OTHER Exception#########################*/