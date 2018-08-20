#ifndef _UMAEXCEPTION_
#define _UMAEXCEPTION_

#include "Global.h"
#include "Logger.h"
#include <stdexcept>

using namespace std;

/*
UMA Exception class
*/
class UMAException: public std::runtime_error {
protected:
	string _errorMessage;
	bool _isFatal;
	Logger *_log;

public:
	UMAException();
	UMAException(string message);
	UMAException(string message, bool isFatal);
	UMAException(string message, bool isFatal, Logger *log);
	virtual string getErrorMessage() const;
	const bool isFatal() const;
	~UMAException();
};

class UMAInternalException : public UMAException {
public:
	UMAInternalException(string message, bool isFatal=true, Logger *log=nullptr);
	virtual string getErrorMessage() const;
	~UMAInternalException();
};

class UMAInvalidArgsException : public UMAException {
public:
	UMAInvalidArgsException(string message, bool isFatal=false, Logger *log = nullptr);
	virtual string getErrorMessage() const;
	~UMAInvalidArgsException();
};

class UMANoResourceException : public UMAException {
public:
	UMANoResourceException(string message, bool isFatal=false, Logger *log = nullptr);
	virtual string getErrorMessage() const;
	~UMANoResourceException();
};

class UMADuplicationException : public UMAException {
public:
	UMADuplicationException(string message, bool isFatal=false, Logger *log = nullptr);
	virtual string getErrorMessage() const;
	~UMADuplicationException();
};

class UMABadOperationException : public UMAException {
public:
	UMABadOperationException(string message, bool isFatal = false, Logger *log = nullptr);
	virtual string getErrorMessage() const;
	~UMABadOperationException();
};

#endif