#include "UMACoreObject.h"
#include "PropertyMap.h"

UMACoreObject::UMACoreObject(const string &uuid, UMACoreConstant::UMA_OBJECT objType, UMACoreObject *parent):
	_uuid(uuid), _objType(objType), _parent(parent) {
	if (_parent) {
		this->_ppm = new PropertyMap(*(_parent->_ppm));
	}
	else {
		this->_ppm = new PropertyMap();
	}
}

UMACoreObject::~UMACoreObject() {
	_parent = nullptr;
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