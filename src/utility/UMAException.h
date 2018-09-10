#ifndef _UMAEXCEPTION_
#define _UMAEXCEPTION_

#include "Global.h"

using namespace std;

class Logger;

/*
UMA Exception class
*/
class UMAException: public std::runtime_error {
public:
	enum UMAExceptionType { UMA_UNKNOWN, UMA_INTERNAL, UMA_INVALID_ARGS, UMA_NO_RESOURCE, UMA_BAD_OPERATION, UMA_DUPLICATION };

protected:
	string _errorMessage;
	bool _isFatal;
	Logger *_log;
	const UMAExceptionType _type;

public:
	UMAException();
	UMAException(string message, UMAExceptionType type);
	UMAException(string message, bool isFatal, UMAExceptionType type);
	UMAException(string message, bool isFatal, Logger *log, const string &ancestors, UMAExceptionType type);
	virtual string getErrorMessage() const;
	const UMAExceptionType getType() const;
	const bool isFatal() const;
	bool isErrorLogged();
	~UMAException();
};

class UMAInternalException : public UMAException {
public:
	UMAInternalException(string message, bool isFatal=true, Logger *log=nullptr, const string &ancestors="");
	virtual string getErrorMessage() const;
	~UMAInternalException();
};

class UMAInvalidArgsException : public UMAException {
public:
	UMAInvalidArgsException(string message, bool isFatal=false, Logger *log = nullptr, const string &ancestors="");
	virtual string getErrorMessage() const;
	~UMAInvalidArgsException();
};

class UMANoResourceException : public UMAException {
public:
	UMANoResourceException(string message, bool isFatal=false, Logger *log = nullptr, const string &ancestors="");
	virtual string getErrorMessage() const;
	~UMANoResourceException();
};

class UMADuplicationException : public UMAException {
public:
	UMADuplicationException(string message, bool isFatal=false, Logger *log = nullptr, const string &ancestors="");
	virtual string getErrorMessage() const;
	~UMADuplicationException();
};

class UMABadOperationException : public UMAException {
public:
	UMABadOperationException(string message, bool isFatal = false, Logger *log = nullptr, const string &ancestors="");
	virtual string getErrorMessage() const;
	~UMABadOperationException();
};

#endif