#ifndef _UMACOREOBJECT_
#define _UMACOREOBJECT_

#include "Global.h"
#include "UMACoreConstant.h"
using namespace std;
using namespace UMACoreConstant;

/*
This is the base class of all UMACore objects, the class itself cannot be instantiated
*/
class UMACoreObject {
protected:
	// the unique id of the object
	const string _uuid;
	// the UMA Object type
	const int _objType;
	// the parent of the current object
	UMACoreObject *_parent;
	// the object's child type and count
	std::map<string, int> _children;

public:
	UMACoreObject(const string &uuid, UMACoreConstant::UMA_OBJECT ObjType, UMACoreObject *parent);
	virtual ~UMACoreObject();
};

#endif