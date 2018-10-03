#include "UMACoreObject.h"
#include "PropertyMap.h"
#include "PropertyPage.h"
#include "CoreService.h"
#include "UMAException.h"

/*
This is the base UMACore object, that every other UMACore object should extend
*/
UMACoreObject::UMACoreObject(const string &uuid, UMACoreConstant::UMA_OBJECT objType, UMACoreObject *parent):
	_uuid(uuid), _objType(objType), _parent(parent) {
	// All UMACore object should automatically inherit whatever is from its parent
	if (_parent) {
		this->_ppm = new PropertyMap(*(_parent->_ppm));
	}
	else {
		// if the parent does not exist, then ppm is initiated to be null
		this->_ppm = new PropertyMap();
	}

	// then based on the type of UMACore object, just read the property map and layer it in
	string objName = UMACoreConstant::getUMAObjName(objType);
	PropertyMap *pm = CoreService::instance()->getPropertyMap(objName);
	if (pm) {
		_ppm->extend(pm);
	}
}

UMACoreObject::~UMACoreObject() {
	delete this->_ppm;
	_parent = nullptr;
}

PropertyMap *UMACoreObject::getPropertyMap() const {
	return _ppm;
}

const string UMACoreObject::getParentChain() {
	if (_ancestors.empty()) {
		const UMACoreObject *current = this;
		while (current) {
			_ancestors.push_back(current);
			current = current->_parent;
		}
	}
	
	string result = _uuid;
	for (int i = 1; i < _ancestors.size(); ++i) {
		result = _ancestors[i]->_uuid + ":" + result;
	}
	result = "[" + result + "]";
	return result;
}