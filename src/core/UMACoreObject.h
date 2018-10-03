#ifndef _UMACOREOBJECT_
#define _UMACOREOBJECT_

#include "Global.h"
#include "UMACoreConstant.h"
using namespace std;
using namespace UMACoreConstant;

class PropertyMap;

/*
This is the base class of all UMACore objects, the class itself cannot be instantiated
*/
class DLL_PUBLIC UMACoreObject {
protected:
	// the unique id of the object
	const string _uuid;
	// the UMA Object type
	const UMACoreConstant::UMA_OBJECT _objType;
	// the parent of the current object
	UMACoreObject *_parent;
	// the object's child type and count
	std::map<UMACoreConstant::UMA_OBJECT, int> _children;
	// the ancestors of the UMA object, and the variable is only created on demand
	vector<const UMACoreObject*> _ancestors;
	// the property map of an uma obj, value is inherited from parent automatically
	PropertyMap *_ppm;

protected:
	const string getParentChain();

public:
	UMACoreObject(const string &uuid, UMACoreConstant::UMA_OBJECT ObjType, UMACoreObject *parent);
	virtual ~UMACoreObject();
	PropertyMap *getPropertyMap() const;
};

#endif