#include "UMAException.h"

UMAException::UMAException(std::string message, int error_level, http::status_code error_code) : std::runtime_error(message) {
	_error_level = error_level;
	_error_message = message;
	_error_code = error_code;
}

UMAException::UMAException(std::string message, int error_level) : std::runtime_error(message) {
	_error_level = error_level;
	_error_message = message;
}

UMAException::UMAException():std::runtime_error("") {}

string UMAException::getErrorMessage() {
	return _error_message;
}

http::status_code UMAException::getErrorCode() {
	return _error_code;
}

int UMAException::getErrorLevel() {
	return _error_level;
}

CoreException::CoreException(std::string message, int error_level, http::status_code error_code) : UMAException(message, error_level, error_code) {}

CoreException::CoreException(std::string message, int error_level) : UMAException(message, error_level) {}

ServerException::ServerException(std::string message, int error_level, http::status_code error_code) : UMAException(message, error_level, error_code) {}

ServerException::ServerException(std::string message, int error_level) : UMAException(message, error_level) {}

ClientException::ClientException(std::string message, int error_level, const http::status_code error_code) : UMAException(message, error_level, error_code) {}

ClientException::ClientException() {}