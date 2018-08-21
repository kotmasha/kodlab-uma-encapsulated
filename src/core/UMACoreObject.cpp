#include "UMACoreObject.h"

UMACoreObject::UMACoreObject(const string &uuid, UMACoreConstant::UMA_OBJECT objType, UMACoreObject *parent):
	_uuid(uuid), _objType(objType), _parent(parent) {
}


UMACoreObject::~UMACoreObject() {
	_parent = nullptr;
}