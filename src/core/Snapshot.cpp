#include "Snapshot.h"
#include "World.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "AttrSensor.h"
#include "AttrSensorPair.h"
#include "Logger.h"
#include "DataManager.h"
#include "UMAException.h"
#include "UMAutil.h"
#include "CoreService.h"
#include "PropertyMap.h"
#include "PropertyPage.h"

/*
----------------Snapshot Base Class-------------------
*/
extern int ind(int row, int col);
extern int compi(int x);
extern bool qless(double d1, double d2);
static Logger snapshotLogger("Snapshot", "log/snapshot.log");

Snapshot::Snapshot(const string &uuid, UMACoreObject *parent, UMA_SNAPSHOT type) : UMACoreObject(uuid, UMA_OBJECT::SNAPSHOT, parent), _type(type) {
	layerInConf();
	
	_total = stod(_ppm->get("total"));
	_total_ = _total;
	snapshotLogger.debug("Setting init total value to " + to_string(_total), this->getParentChain());
	_delayCount = 0;
	
	_q = 0;
	snapshotLogger.debug("Setting q value to " + to_string(_q), this->getParentChain());

	_threshold = stod(_ppm->get("threshold"));
	snapshotLogger.debug("Setting threshold value to " + to_string(_threshold), this->getParentChain());

	_autoTarget = stoi(_ppm->get("auto_target"));
	snapshotLogger.debug("Setting auto target value to " + to_string(_autoTarget), this->getParentChain());

	_propagateMask = stoi(_ppm->get("propagate_mask"));
	snapshotLogger.debug("Setting propagate mask value to " + to_string(_propagateMask), this->getParentChain());

	_initialSize = 0;

	_dm = new DataManager(this);
	snapshotLogger.info("Data Manager is created", this->getParentChain());

	snapshotLogger.info("A Snapshot " + _uuid + " is created, with type " + to_string(_type), this->getParentChain());
}


Snapshot::~Snapshot(){
	try {
		for (int i = 0; i < _sensors.size(); ++i) {
			delete _sensors[i];
			_sensors[i] = NULL;
		}
		snapshotLogger.debug("All snapshot sensors are deleted", this->getParentChain());
		for (int i = 0; i < _sensorPairs.size(); ++i) {
			delete _sensorPairs[i];
			_sensorPairs[i] = NULL;
		}
		snapshotLogger.debug("All snapshot sensor pairs are deleted", this->getParentChain());
		delete _dm;
		snapshotLogger.debug("Data Manager is deleted", this->getParentChain());
	}
	catch (exception &e) {
		throw UMAInternalException("Fatal error in Snapshot destruction function, snapshotId=" + _uuid, true, &snapshotLogger, this->getParentChain());
	}
	snapshotLogger.info("Snapshot is deleted, snapshotId=" + _uuid, this->getParentChain());
}

Sensor *Snapshot::createSensor(const std::pair<string, string> &idPair, const vector<double> &diag, const vector<vector<double> > &w, const vector<vector<bool> > &b) {
	_dm->copyArraysToSensors(0, _sensors.size(), _sensors);
	_dm->copyArraysToSensorPairs(0, _sensors.size(), _sensorPairs);
	if (_sensorIdx.find(idPair.first) != _sensorIdx.end() && _sensorIdx.find(idPair.second) != _sensorIdx.end()) {
		throw UMADuplicationException("Cannot create a duplicate sensor, sensorId=[" + idPair.first + ", " + idPair.second + "]", false, &snapshotLogger, this->getParentChain());
	}
	Sensor *sensor = NULL;
	if (diag.empty()) {
		sensor = new Sensor(idPair, this, _total, _sensors.size());
	}
	else {
		sensor = new Sensor(idPair, this, diag, _sensors.size());
	}
	_sensorIdx[idPair.first] = sensor;
	_sensorIdx[idPair.second] = sensor;
	_sensors.push_back(sensor);
	snapshotLogger.debug("A Sensor is created, sensorId=" + idPair.first + ", sensorIdx=" + to_string(sensor->_idx), this->getParentChain());
	//creating sensor pairs
	for (int i = 0; i < _sensors.size(); ++i) {
		SensorPair *sensorPair = NULL;
		if (UMA_SNAPSHOT::SNAPSHOT_QUALITATIVE == _type) {
			sensorPair = new SensorPair(this, sensor, _sensors[i], _threshold);
		}
		else {
			if (w.empty()) {
				sensorPair = new SensorPair(this, sensor, _sensors[i], _threshold, _total);
			}
			else {
				sensorPair = new SensorPair(this, sensor, _sensors[i], _threshold, w[i], b[i]);
			}
		}
		snapshotLogger.debug("A sensor pair with is created, sensor1=" + sensor->_uuid + " sensor2=" + _sensors[i]->_uuid, this->getParentChain());
		_sensorPairs.push_back(sensorPair);
	}
	snapshotLogger.info(to_string(_sensors.size()) + " Sensor Pairs are created, total is " + to_string(ind(_sensors.size(), 0)), this->getParentChain());
	
	if (_sensors.size() > _dm->_sensorSizeMax) {
		snapshotLogger.debug("Need allocate more space after adding a sensor", this->getParentChain());
		_dm->reallocateMemory(_total, _sensors.size());
		_dm->createSensorsToArraysIndex(0, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(0, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(0, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(0, _sensors.size(), _sensorPairs);
	}
	else {
		snapshotLogger.debug("Have enough space, will not do remalloc", this->getParentChain());
		_dm->setSize(_sensors.size(), false);
		_dm->createSensorsToArraysIndex(_sensors.size() - 1, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(_sensors.size() - 1, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(_sensors.size() - 1, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(_sensors.size() - 1, _sensors.size(), _sensorPairs);
	}
	return sensor;
}

void Snapshot::deleteSensor(const string &sensorId) {
	if (_sensorIdx.find(sensorId) == _sensorIdx.end()) {
		throw UMANoResourceException("Cannot find the sensor, sensorId=" + sensorId, false, &snapshotLogger, this->getParentChain());
	}

	int sensor_idx = _sensorIdx.at(sensorId)->_idx;
	vector<bool> pruningList(_dm->_attrSensorSize, false);
	pruningList[2 * sensor_idx] = true;
	pruning(pruningList);
	snapshotLogger.info("Sensor is deleted, sensorId=" + sensorId, this->getParentChain());
}

/*
This function is adding a dup sensor to the snapshot, and copy the important value to it
*/
Sensor *Snapshot::addSensor(Sensor * const sensor) {
	std::pair<string, string> idPairs = { sensor->_m->_uuid, sensor->_cm->_uuid };
	vector<double> diag;
	vector<vector<double>> w;
	vector<vector<bool>> b;

	Sensor *newSensor = createSensor(idPairs, diag, w, b);
	newSensor->mergeSensor(sensor);

	return newSensor;
}

vector<vector<string> > Snapshot::getSensorInfo() const {
	vector<vector<string>> results;
	for (int i = 0; i < _sensors.size(); ++i) {
		Sensor * const sensor = _sensors[i];
		vector<string> sensorPair = { _sensors[i]->_m->_uuid, _sensors[i]->_cm->_uuid };
		results.push_back(sensorPair);
	}
	return results;
}

/*
This function is merging snapshot with the current snapshot, and the merging rule is like this:
1 only copy sensor from snapshot, that exist in current snapshot
2 for the sensor that is copied, go through the amper list, and remove any sensor that is not in the list from current snapshot
*/
void Snapshot::mergeSnapshot(Snapshot * const snapshot) {
	// 1st copy all initial sensors from snapshot to current snapshot, if current snapshot has them.
	// only copy with basic info
	// otherwise, prune the sensor in snapshot
	vector<string> mids;
	for (int i = 0; i < snapshot->_initialSize; ++i) {
		Sensor * const sensor = snapshot->_sensors[i];
		if (_sensorIdx.end() != _sensorIdx.find(sensor->_uuid)) {
			// find the sensors, and copy them
			Sensor * currentSensor = _sensorIdx[sensor->_uuid];
			currentSensor->mergeSensor(sensor);
		}
		else {
			// not find the sensor, prune them in snaphsot, but get the attrSensor ids here
			mids.push_back(sensor->_m->_uuid);
		}
	}
	vector<bool> sig = snapshot->generateSignal(mids);
	snapshot->pruning(sig);
	// after pruning, the snapshot will only remain initial sensor that all exist in current snapshot
	// all other delayed sensor will also be related to those initial sensor, so we copy them directly to current sensor

	// 2nd just copy all delayed sensor from snapshot to current snapshot
	for (int i = snapshot->_initialSize; i < snapshot->_sensors.size(); ++i) {
		Sensor * const sensor = snapshot->_sensors[i];
		addSensor(sensor);
		// after adding sensor, the new sensor in current snapshot will have correct index
	}

	// 3rd copy data in sensor pairs
	for (int i = 0; i < snapshot->_sensorPairs.size(); ++i) {
		SensorPair * const sensorPair = snapshot->_sensorPairs[i];
		
		Sensor *s1 = _sensorIdx[sensorPair->_sensor_i->_uuid], *s2 = _sensorIdx[sensorPair->_sensor_j->_uuid];
		SensorPair *currentSensorPair = getSensorPair(s1, s2);

		currentSensorPair->mergeSensorPair(sensorPair);
	}
	//TODO, check for _sensorIdx, hash, amper, and other stuff
}

/*
This function is copying the snapshot data to the current one, but only copying the sensor/sensorpair that already exist in current snapshot
*/
/*
void Snapshot::copy_test_data(Snapshot *snapshot) {
	for (int i = 0; i < _sensor_num; ++i) {
		//the current sensor to look at
		Sensor *sensor = _sensors[i];
		//the current sensor id
		string sensor_id = sensor->_uuid;
		if (snapshot->_sensorIdx.find(sensor_id) != snapshot->_sensorIdx.end()) {
			//if the current sensor id also in the other snapshot

			//copy sensors
			//get the 'same' sensor in the other snapshot
			Sensor *c_sensor = snapshot->_sensorIdx[sensor_id];
			//copy the value
			sensor->copy_data(c_sensor);
			//reconstruct the amper list
			vector<int> amper_list;
			for (int i = 0; i < c_sensor->_amper.size(); ++i) {
				int idx = c_sensor->_amper[i];
				bool isOriginPure = (idx % 2 == 0);
				Sensor *a_sensor = _sensors[idx / 2];
				if (_sensorIdx.find(a_sensor->_uuid) != _sensorIdx.end()) {
					if (isOriginPure) {
						amper_list.push_back(_sensorIdx[a_sensor->_uuid]->_m->_idx);
						_log->debug() << "found the amper sensor(" + a_sensor->_uuid + ") in Pure position in new test";
					}
					else {
						amper_list.push_back(_sensorIdx[a_sensor->_uuid]->_cm->_idx);
						_log->debug() << "found the amper sensor(" + a_sensor->_uuid + ") in Complementary position in new test";
					}
				}
				else {
					_log->debug() << "Cannot found the amper sensor(" + a_sensor->_uuid + ") in new test";
				}
			}
			//set the amper list
			sensor->setAmperList(amper_list);
			_log->debug() << "Sensor(" + sensor_id + ") data are copied";

			//copy sensor pairs
			for (int j = 0; j <= i; ++j) {
				//go through all the sensor pair list of the current sensor
				//the current other sensor
				Sensor *s = _sensors[j];
				if (snapshot->_sensorIdx.find(s->_uuid) != snapshot->_sensorIdx.end()) {
					//if the other sensor id also in the other snapshot
					//get the current sensor pair
					SensorPair *sensor_pair = _sensorPairs[ind(i, j)];
					//get the other 'same' sensor
					Sensor *c_s = snapshot->_sensorIdx[s->_uuid];
					//get the corresponding sensorpair in the other snapshot, the idx of the sensorpair in the other one maybe different
					SensorPair *c_sensor_pair = snapshot->getSensorPair(c_sensor, c_s);
					//copy the data
					sensor_pair->copy_data(c_sensor_pair);
					_log->debug() << "Sensor Pair(sensor1=" + sensor_id + ", sensor2=" + s->_uuid + ") are copied";
				}
				else {
					_log->debug() << "Cannot find the sensor(" + s->_uuid + ") while copying sensor pair, must be a new sensor";
				}
			}//j
		}//if
		else {
			_log->debug() << "Cannot find the sensor(" + sensor_id + ") while copying sensor, must be a new sensor";
		}
	}//i
}
*/

/*
The function is pruning the sensor and sensor pair list, also adjust their corresponding idx
Input: the signal of all attr_sensor
*/
void Snapshot::pruning(const vector<bool> &signal){
	if (signal.size() > _dm->_attrSensorSize) {
		throw UMAInvalidArgsException("Input signal size for pruning is larger than attrSensorSize, inputSize="
			+ to_string(signal.size()) + ", attrSensorSize=" + to_string(_dm->_attrSensorSize), false, &snapshotLogger, this->getParentChain());
	}
	//get converted sensor list, from attr_sensor signal
	const vector<bool> sensorList = SignalUtil::attrSensorToSensorSignal(signal);
	const vector<int> idxList = SignalUtil::boolSignalToIntIdx(sensorList);
	if (idxList.empty()) {
		snapshotLogger.info("Empty pruning signal, do nothing", this->getParentChain());
		return;
	}
	_dm->copyArraysToSensors(0, _sensors.size(), _sensors);
	_dm->copyArraysToSensorPairs(0, _sensors.size(), _sensorPairs);

	if (idxList[0] < 0 || idxList.back() >= _sensors.size()) {
		throw UMABadOperationException("Pruning range is from " + to_string(idxList[0]) + "~" + to_string(idxList.back())
			+ ", illegal range!", false, &snapshotLogger, this->getParentChain());
	}

	string strList = "";
	int prunedInitialSensorCount = 0;
	for (int i = 0; i < idxList.size(); ++i) {
		strList += to_string(idxList[i]) + ", ";
		if (idxList[i] < _initialSize) ++prunedInitialSensorCount;
	}
	snapshotLogger.info("Will prune id=" + strList, this->getParentChain());

	int rowEscape = 0;
	int totalEscape = 0;
	//destruct the corresponding sensors and sensor pairs
	for(int i = 0; i < _sensors.size(); ++i){
		if(sensorList[i]){
			//delete the sensor if necessary
			_sensorIdx.erase(_sensors[i]->_m->_uuid);
			_sensorIdx.erase(_sensors[i]->_cm->_uuid);
			vector<int> amper_list = _sensors[i]->getAmperList();
			vector<bool> amper_signal = SignalUtil::intIdxToBoolSignal(amper_list, _sensors.size() * 2);
			size_t delay_list_hash = delayHash(SignalUtil::trimSignal(amper_signal));
			_delaySensorHash.erase(delay_list_hash);

			delete _sensors[i];
			_sensors[i] = NULL;
			rowEscape++;
		}
		else{
			//or just adjust the idx of the sensor, and change the position
			vector<int> amperList = _sensors[i]->getAmperList();
			vector<int> newAmperList;
			for (int j = 0; j < amperList.size(); ++j) {
				int idx = ArrayUtil::findIdxInSortedArray(idxList, amperList[j] / 2);
				if (idx >= 0 && idxList[idx] != amperList[j] / 2) {//if the value is not to be pruned
					newAmperList.push_back(amperList[j] - 2 * (idx + 1));
				}
			}

			size_t oldDelayHash = delayHash(SignalUtil::trimSignal(SignalUtil::intIdxToBoolSignal(amperList, _sensors.size() * 2)));
			size_t newDelayHash = delayHash(SignalUtil::trimSignal(SignalUtil::intIdxToBoolSignal(newAmperList, _sensors.size() * 2)));
			_delaySensorHash.erase(oldDelayHash);
			_delaySensorHash.insert(newDelayHash);

			_sensors[i]->_amper.clear();
			_sensors[i]->_amper = newAmperList;
			_sensors[i]->setIdx(i - rowEscape);
			_sensors[i - rowEscape] = _sensors[i];
		}
		//delete the row of sensor, or the col in a row, where ther other sensor is deleted
		for(int j = 0; j <= i; ++j){
			if(sensorList[i] || sensorList[j]){
				//delete the sensor pair if necessary
				delete _sensorPairs[ind(i, j)];
				_sensorPairs[ind(i, j)] = NULL;
				totalEscape++;
			}
			else{
				//or just change the position
				_sensorPairs[ind(i, j) - totalEscape] = _sensorPairs[ind(i, j)];
			}
		}
	}

	//earse the additional space
	_sensors.erase(_sensors.end() - rowEscape, _sensors.end());
	_sensorPairs.erase(_sensorPairs.end() - totalEscape, _sensorPairs.end());
	//adjust the size variables
	_dm->setSize(_sensors.size(), false);

	_dm->createSensorsToArraysIndex(0, _sensors.size(), _sensors);
	_dm->createSensorPairsToArraysIndex(0, _sensors.size(), _sensorPairs);
	_dm->copySensorsToArrays(0, _sensors.size(), _sensors);
	_dm->copySensorPairsToArrays(0, _sensors.size(), _sensorPairs);

	setInitialSize(_initialSize - prunedInitialSensorCount);

	snapshotLogger.info("Pruning done successful, snapshot_id=" + _uuid, this->getParentChain());
}

void Snapshot::ampers(const vector<vector<bool> > &lists, const vector<std::pair<string, string> > &idPairs){
	_dm->copyArraysToSensors(0, _sensors.size(), _sensors);
	_dm->copyArraysToSensorPairs(0, _sensors.size(), _sensorPairs);
	int successAmper = 0;
	//record how many delay are successful
	
	for(int i = 0; i < lists.size(); ++i){
		const vector<int> list = SignalUtil::boolSignalToIntIdx(lists[i]);
		if (list.size() < 2) {
			snapshotLogger.warn("The amper vector size is less than 2, will abort this amper operation, snapshotId=" + _uuid + " listId=" + to_string(i), this->getParentChain());
			continue;
		}
		amper(list, idPairs[i]);
		successAmper++;
	}

	snapshotLogger.info(to_string(successAmper) + " out of " + to_string(lists.size()) + " amper successfully done, snapshotId=" + _uuid, this->getParentChain());

	if(_sensors.size() > _dm->_sensorSizeMax){
		snapshotLogger.info("New sensor size larger than current max, will resize", this->getParentChain());
		//if need to reallocate
		_dm->reallocateMemory(_total, _sensors.size());
		//copy every sensor back, since the memory is new
		_dm->createSensorsToArraysIndex(0, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(0, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(0, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(0, _sensors.size(), _sensorPairs);
	}
	else{
		//else just update the actual size
		_dm->setSize(_sensors.size(), false);
		//copy just the new added sensors and sensor pairs
		_dm->createSensorsToArraysIndex(_sensors.size() - successAmper, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(_sensors.size() - successAmper, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(_sensors.size() - successAmper, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(_sensors.size() - successAmper, _sensors.size(), _sensorPairs);
	}
}


void Snapshot::delays(const vector<vector<bool> > &lists, const vector<std::pair<string, string> > &idPairs) {
	_dm->copyArraysToSensors(0, _sensors.size(), _sensors);
	_dm->copyArraysToSensorPairs(0, _sensors.size(), _sensorPairs);

	bool generateDefaultId = idPairs.empty();
	pair<string, string> p;
	int successDelay = 0;
	//record how many delay are successful
	for (int i = 0; i < lists.size(); ++i) {
		size_t v = delayHash(SignalUtil::trimSignal(lists[i]));
		if (_delaySensorHash.end() != _delaySensorHash.find(v)) {
			snapshotLogger.info("Find an existing delayed sensor, will skip creating current one, snapshotId=" + _uuid, this->getParentChain());
			continue;
		}

		if (lists[i].size() > _sensors.size() * 2) {
			throw UMAInvalidArgsException("The " + to_string(i) + "th input signal size is larger than 2 * sensors.size()", false, &snapshotLogger, this->getParentChain());
		}
		const vector<int> list = SignalUtil::boolSignalToIntIdx(lists[i]);
		if (list.size() < 1) {
			snapshotLogger.warn("The amper vector size is less than 1, will abort this amper operation, list id: " + to_string(i), this->getParentChain());
			continue;
		}
		if(generateDefaultId){
			int delayCount = getDelayCount();
			p = { "delay" + to_string(delayCount), "c_delay" + to_string(delayCount) };
		}
		else {
			p = idPairs[i];
		}
		
		if (list.size() == 1) {
			try {
				generateDelayedWeights(list[0], true, p);
			}
			catch (UMAException &e) {
				throw UMAInternalException("Fatal error in generateDelayedWeights, snapshotId=" + _uuid, true, &snapshotLogger, this->getParentChain());
			}
		}
		else {
			amper(list, p);
			try {
				generateDelayedWeights(_sensors.back()->_m->_idx, false, p);
			}
			catch (UMAException &e) {
				throw UMAInternalException("Fatal error in generateDelayedWeights, snapshotId=" + _uuid, true, &snapshotLogger, this->getParentChain());
			}
		}
		successDelay++;
		_delaySensorHash.insert(v);

		snapshotLogger.info("A delayed sensor is generated, sensorId=" + p.first + ", snapshotId=" + _uuid, this->getParentChain());
		string delay_list = "";
		for (int j = 0; j < list.size(); ++j) delay_list += (to_string(list[j]) + ",");
		snapshotLogger.verbose("The delayed sensor, sensorId= " + _uuid + " is generated from " + delay_list, this->getParentChain());
	}

	snapshotLogger.info(to_string(successDelay) + " out of " + to_string(lists.size()) + " delay successfully done", this->getParentChain());

	if (_sensors.size() > _dm->_sensorSizeMax) {
		snapshotLogger.info("New sensor size larger than current max, will resize", this->getParentChain());
		//if need to reallocate
		_dm->reallocateMemory(_total, _sensors.size());
		//copy every sensor back, since the memory is new
		_dm->createSensorsToArraysIndex(0, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(0, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(0, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(0, _sensors.size(), _sensorPairs);
	}
	else {
		//else just update the actual size
		_dm->setSize(_sensors.size(), false);
		//copy just the new added sensors and sensor pairs
		_dm->createSensorsToArraysIndex(_sensors.size() - successDelay, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(_sensors.size() - successDelay, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(_sensors.size() - successDelay, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(_sensors.size() - successDelay, _sensors.size(), _sensorPairs);
	}
}

void Snapshot::layerInConf() {
	string confName = "Snapshot::" + UMACoreConstant::getUMASnapshotName(_type);
	PropertyMap *pm = CoreService::instance()->getPropertyMap(confName);
	if (pm) {
		_ppm->extend(pm);
	}
}

//####################################################################################
//------------------------------------SET FUNCTION------------------------------------

void Snapshot::setThreshold(const double &threshold) {
	_threshold = threshold;
	snapshotLogger.info("snapshot threshold changed to " + to_string(threshold), this->getParentChain());
}

void Snapshot::setQ(const double &q) {
	_q = q;
	snapshotLogger.info("snapshot q changed to " + to_string(q), this->getParentChain());
}

void Snapshot::setAutoTarget(const bool &autoTarget) {
	_autoTarget = autoTarget;
	snapshotLogger.info("snapshot auto target to " + to_string(_autoTarget));
}

void Snapshot::setPropagateMask(const bool &propagateMask) {
	_propagateMask = propagateMask;
	snapshotLogger.info("snapshot propagate mask to " + to_string(_propagateMask));
}

void Snapshot::setInitialSize(const int &initialSize) {
	_initialSize = initialSize;
	snapshotLogger.info("snapshot initial size to " + to_string(_initialSize));
}

void Snapshot::setInitialSize() {
	_initialSize = _sensors.size();
}

void Snapshot::setTotal(const double &total){
	_total = total;
	snapshotLogger.info("snapshot total to " + to_string(_total));
}

void Snapshot::setOldTotal(const double &total_) {
	_total_ = total_;
	snapshotLogger.info("snapshot old total to " + to_string(_total_));
}


//------------------------------------SET FUNCTION------------------------------------
//####################################################################################

//####################################################################################
//------------------------------------GET FUNCTION------------------------------------

/*
this function is getting the attr_sensor, from the sensor list
*/
AttrSensor *Snapshot::getAttrSensor(int idx){
	int sIdx = idx / 2;
	if (sIdx >= _sensors.size() || sIdx <0) {
		throw UMAInvalidArgsException("the input attr_sensor index is out of range, input is " + to_string(sIdx) +
			" sensor num is " + to_string(idx), false, &snapshotLogger, this->getParentChain());
	}
	if(idx % 2 == 0){
		return _sensors[sIdx]->_m;
	}
	else{
		return _sensors[sIdx]->_cm;
	}
}

AttrSensor *Snapshot::getAttrSensor(const string &attrSensorId){
	Sensor *sensor = getSensor(attrSensorId);
	if (attrSensorId == sensor->_m->_uuid) return sensor->_m;
	else if(attrSensorId == sensor->_cm->_uuid) return sensor->_cm;

	throw UMANoResourceException("Cannot find the attr_sensor id " + attrSensorId, false, &snapshotLogger, this->getParentChain());
}

SensorPair *Snapshot::getSensorPair(const Sensor *sensor1, const Sensor *sensor2){
	int idx1 = sensor1->_idx > sensor2->_idx ? sensor1->_idx : sensor2->_idx;
	int idx2 = sensor1->_idx > sensor2->_idx ? sensor2->_idx : sensor1->_idx;
	return _sensorPairs[ind(idx1, idx2)];
}

/*
This function is getting the attr_sensor pair from the sensor pair list
Input: m_idx1, m_idx2 are index of the attr_sensor, m_idx1 > m_idx2
*/
AttrSensorPair *Snapshot::getAttrSensorPair(int mIdx1, int mIdx2){
	int idx1 = mIdx1 > mIdx2 ? mIdx1 : mIdx2;
	int idx2 = mIdx1 > mIdx2 ? mIdx2 : mIdx1;
	int sIdx1 = idx1 / 2;
	int sIdx2 = idx2 / 2;
	AttrSensor *m1 = getAttrSensor(idx1);
	AttrSensor *m2 = getAttrSensor(idx2);
	return _sensorPairs[ind(sIdx1, sIdx2)]->getAttrSensorPair(m1->_isOriginPure, m2->_isOriginPure);
}

AttrSensorPair *Snapshot::getAttrSensorPair(const string &mid1, const string &mid2){
	int idx1 = getAttrSensor(mid1)->_idx;
	int idx2 = getAttrSensor(mid2)->_idx;
	return getAttrSensorPair(idx1, idx2);
}

vector<bool> Snapshot::getAmperList(const string &sensorId){
	if (_sensorIdx.find(sensorId) == _sensorIdx.end()) {
		throw UMANoResourceException("Cannot find the sensor id " + sensorId, false, &snapshotLogger, this->getParentChain());
	}
	Sensor *sensor = _sensorIdx.at(sensorId);
	vector<bool> result(_dm->_attrSensorSize, false);
	for (int i = 0; i < sensor->_amper.size(); ++i) {
		result[sensor->_amper[i]] = true;
	}
	return result;
}

vector<string> Snapshot::getAmperListID(const string &sensorId){
	if (_sensorIdx.find(sensorId) == _sensorIdx.end()) {
		throw UMANoResourceException("Cannot find the sensor id " + sensorId, false, &snapshotLogger, this->getParentChain());
	}
	Sensor * const sensor = _sensorIdx.at(sensorId);
	vector<string> result;
	for (int i = 0; i < sensor->_amper.size(); ++i) {
		int idx = sensor->_amper[i];
		Sensor *s = _sensors[idx / 2];
		if (idx % 2 == 0) result.push_back(s->_m->_uuid);
		else result.push_back(s->_cm->_uuid);
	}
	return result;
}

Sensor *Snapshot::getSensor(const string &sensorId){
	if (_sensorIdx.find(sensorId) == _sensorIdx.end()) {
		throw UMANoResourceException("Cannot find the sensor id " + sensorId, false, &snapshotLogger, this->getParentChain());
	}
	return _sensorIdx.at(sensorId);
}

const double &Snapshot::getTotal() const{
	return _total;
}

const double &Snapshot::getOldTotal() const{
	return _total_;
}

const double &Snapshot::getQ() const{
	return _q;
}

const double &Snapshot::getThreshold() const{
	return _threshold;
}

const bool &Snapshot::getAutoTarget() const{
	return _autoTarget;
}

const bool &Snapshot::getPropagateMask() const{
	return _propagateMask;
}

const int &Snapshot::getInitialSize() const{
	return _initialSize;
}

const UMA_SNAPSHOT &Snapshot::getType() const {
	return _type;
}

const int Snapshot::getDelayCount() {
	return _delayCount++;
}

DataManager *Snapshot::getDM() const {
	return _dm;
}

//------------------------------------GET FUNCTION------------------------------------
//####################################################################################

//the input list size should be larger than 2
void Snapshot::amper(const vector<int> &list, const std::pair<string, string> &uuid) {
	if (list.size() < 2) {
		snapshotLogger.warn("Amper list size is smaller than 2, will not continue", this->getParentChain());
		return;
	}
	try {
		ampersand(list[1], list[0], true, uuid);
		for (int j = 2; j < list.size(); ++j) {
			ampersand(_sensors.back()->_m->_idx, list[j], false, uuid);
		}
	}
	catch (UMAException &e) {
		throw UMAInternalException("Fatal error while doing ampersand", true, &snapshotLogger, this->getParentChain());
	}
}

/*
This is the amper and function, used in amper and delay
Input: m_idx1, m_idx2 are the attr_sensor idx that need to be ampersand, m_idx1 > m_idx2, merge is indicating whether merge or replace the last sensor/row of sensor pair
*/
void Snapshot::ampersand(int mIdx1, int mIdx2, bool merge, const std::pair<string, string> &idPair) {
	vector<SensorPair*> amperAndSensorPairs;
	Sensor *amperAndSensor = new Sensor(idPair, this, _total, _sensors.size());
	_sensorIdx[idPair.first] = amperAndSensor;
	_sensorIdx[idPair.second] = amperAndSensor;

	double f = 1.0;
	if(_total > 1e-12)
		f = getAttrSensorPair(mIdx1, mIdx2)->_vw / _total;
	for(int i = 0; i < _sensors.size(); ++i){
		SensorPair *sensorPair = NULL;
		sensorPair = new SensorPair(this, amperAndSensor, _sensors[i], _threshold, _total);
		AttrSensor *m1 = _sensors[i]->_m;
		AttrSensor *m2 = _sensors[i]->_cm;

		if (_sensors[i]->_m->_idx == mIdx1 || _sensors[i]->_m->_idx == mIdx2) {
			sensorPair->mij->_vw = getAttrSensorPair(mIdx1, mIdx2)->_vw;
			sensorPair->mi_j->_vw = 0.0;
			if (_sensors[i]->_m->_idx == mIdx1) {
				sensorPair->m_ij->_vw = getAttrSensorPair(mIdx1, compi(mIdx2))->_vw;
				sensorPair->m_i_j->_vw = getAttrSensorPair(compi(mIdx1), compi(mIdx1))->_vw;
			}
			else {
				sensorPair->m_ij->_vw = getAttrSensorPair(compi(mIdx1), mIdx2)->_vw;
				sensorPair->m_i_j->_vw = getAttrSensorPair(compi(mIdx2), compi(mIdx2))->_vw;
			}
		}
		else if (_sensors[i]->_cm->_idx == mIdx1 || _sensors[i]->_cm->_idx == mIdx2) {
			sensorPair->mi_j->_vw = getAttrSensorPair(mIdx1, mIdx2)->_vw;
			sensorPair->mij->_vw = 0.0;
			if (_sensors[i]->_cm->_idx == mIdx1) {
				sensorPair->m_i_j->_vw = getAttrSensorPair(mIdx1, compi(mIdx2))->_vw;
				sensorPair->m_ij->_vw = getAttrSensorPair(compi(mIdx1), compi(mIdx1))->_vw;
			}
			else {
				sensorPair->m_i_j->_vw = getAttrSensorPair(compi(mIdx1), mIdx2)->_vw;
				sensorPair->m_ij->_vw = getAttrSensorPair(compi(mIdx2), compi(mIdx2))->_vw;
			}
		}
		else {
			sensorPair->mij->_vw = f * getAttrSensorPair(m1->_idx, m1->_idx)->_vw;
			sensorPair->mi_j->_vw = f * getAttrSensorPair(m2->_idx, m2->_idx)->_vw;
			sensorPair->m_ij->_vw = (1.0 - f) * getAttrSensorPair(m1->_idx, m1->_idx)->_vw;
			sensorPair->m_i_j->_vw = (1.0 - f) * getAttrSensorPair(m2->_idx, m2->_idx)->_vw;
		}
		amperAndSensorPairs.push_back(sensorPair);
	}
	SensorPair *selfPair = new SensorPair(this, amperAndSensor, amperAndSensor, _threshold, _total);
	selfPair->mij->_vw = amperAndSensorPairs[0]->mij->_vw + amperAndSensorPairs[0]->mi_j->_vw;
	selfPair->mi_j->_vw = 0.0;
	selfPair->m_ij->_vw = 0.0;
	selfPair->m_i_j->_vw = amperAndSensorPairs[0]->m_ij->_vw + amperAndSensorPairs[0]->m_i_j->_vw;
	amperAndSensorPairs.push_back(selfPair);
	if(!merge){
		Sensor *oldSensor = _sensors.back();
		//restore the old sensor amper list
		amperAndSensor->setAmperList(oldSensor);
		//also append the mIdx2 as the new amper list value
		amperAndSensor->setAmperList(mIdx2);
		//take the idx of the old one, copy it to the new one 
		amperAndSensor->setIdx(oldSensor->_idx);
		//delete the old one
		delete oldSensor;
		//if no mrege needed, means the last existing sensor and row of sensor_pair need to be removed
		_sensors.pop_back();
		//destruct the last row of sensor pair
		for(int i = ind(_sensors.size(), 0); i < ind(_sensors.size() + 1, 0); ++i){
			delete _sensorPairs[i];
		}
		_sensorPairs.erase(_sensorPairs.begin() + ind(_sensors.size(), 0), _sensorPairs.end());
		//also need to remove the n-1 position of amperAndSensorPairs
		delete amperAndSensorPairs[_sensors.size()];
		amperAndSensorPairs.erase(amperAndSensorPairs.end() - 2, amperAndSensorPairs.end() - 1);
	}
	else{
		//set the amper list, if append a new sensor, append the smaller idx first
		amperAndSensor->setAmperList(mIdx2);
		amperAndSensor->setAmperList(mIdx1);
	}
	_sensors.push_back(amperAndSensor);
	_sensorPairs.insert(_sensorPairs.end(), amperAndSensorPairs.begin(), amperAndSensorPairs.end());
}

/*
This function is generating the delayed weights
Before this function is called, have to make sure, _sensors and _sensorPairs have valid info in vwxx;
Input: mid of the attr_sensor doing the delay, and whether to merge after the operation
*/
void Snapshot::generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair){
	//create a new delayed sensor
	Sensor *delayedSensor = new Sensor(idPair, this, _total, _sensors.size());
	_sensorIdx[idPair.first] = delayedSensor;
	_sensorIdx[idPair.second] = delayedSensor;
	vector<SensorPair *> delayedSensorPairs;
	//the sensor name is TBD, need python input
	int sid = mid / 2;
	//get mid and sid

	bool isSensorActive;
	if(merge){
		//if need to merge, means just a single sensor delay
		isSensorActive = _sensors[sid]->getObserve();
	}
	else{
		//means not a single sensor delay
		_sensors[sid]->setObserveList(_dm->h_observe, _dm->h_observe_);
		isSensorActive = _sensors[sid]->generateDelayedSignal();
	}
	//reverse for compi
	if (mid % 2 == 1) isSensorActive = !isSensorActive;

	for(int i = 0; i < _sensors.size() + 1; ++i){
		SensorPair *sensorPair = NULL;
		if(i == _sensors.size()){
			//if this is the last sensor pair, and it is the pair of the delayed sensor itself
			sensorPair = new SensorPair(this, delayedSensor, delayedSensor, _threshold, _total);
			//copy all those diag values first
			delayedSensor->_m->_vdiag = delayedSensorPairs[0]->mij->_vw + delayedSensorPairs[0]->mi_j->_vw;
			delayedSensor->_cm->_vdiag = delayedSensorPairs[0]->m_ij->_vw + delayedSensorPairs[0]->m_i_j->_vw;
			delayedSensor->_m->_vdiag_ = delayedSensor->_m->_vdiag;
			delayedSensor->_cm->_vdiag_ = delayedSensor->_cm->_vdiag;
			//then assign the value to sensor pair
			sensorPair->mij->_vw = delayedSensor->_m->_vdiag;
			sensorPair->mi_j->_vw = 0;
			sensorPair->m_ij->_vw = 0;
			sensorPair->m_i_j->_vw = delayedSensor->_cm->_vdiag;
			delayedSensorPairs.push_back(sensorPair);
		}
		else{
			sensorPair = new SensorPair(this, delayedSensor, _sensors[i], _threshold, _total);
			sensorPair->mij->_vw = isSensorActive * _sensors[i]->_m->_vdiag;
			sensorPair->mi_j->_vw = isSensorActive * _sensors[i]->_cm->_vdiag;
			sensorPair->m_ij->_vw = !isSensorActive * _sensors[i]->_m->_vdiag;
			sensorPair->m_i_j->_vw = !isSensorActive * _sensors[i]->_cm->_vdiag;
			delayedSensorPairs.push_back(sensorPair);
		}
	}

	if(!merge){
		//replace the last one
		Sensor *oldSensor = _sensors[sid];
		//set the mask amper of the delayed sensor to be the same as the one that need to be delayed
		delayedSensor->setAmperList(oldSensor);
		//take the idx of the old one, copy it to the new one 
		delayedSensor->setIdx(oldSensor->_idx);
		//delete the old one
		delete oldSensor;
		//if no mrege needed, means the last existing sensor and row of sensor_pair need to be removed
		_sensors.pop_back();
		//destruct the last row of sensor pair
		for(int i = ind(_sensors.size(), 0); i < ind(_sensors.size() + 1, 0); ++i){
			delete _sensorPairs[i];
		}
		_sensorPairs.erase(_sensorPairs.begin() + ind(_sensors.size(), 0), _sensorPairs.end());
		//also need to remove the n-1 position of delayedSensor_pair
		delete delayedSensorPairs[_sensors.size()];
		delayedSensorPairs.erase(delayedSensorPairs.end() - 2, delayedSensorPairs.end() - 1);
	}
	else{
		delayedSensor->setAmperList(mid);
	}
	_sensors.push_back(delayedSensor);
	_sensorPairs.insert(_sensorPairs.end(), delayedSensorPairs.begin(), delayedSensorPairs.end());
}

void Snapshot::generateObserve(vector<bool> &observe) {
	if (observe.size() != 2 * _initialSize) {
		throw UMAInvalidArgsException("The input observe signal size is not the 2x initial sensor size", false, &snapshotLogger, this->getParentChain());
	}
	for (int i = _initialSize; i < _sensors.size(); ++i) {
		bool b = _sensors[i]->generateDelayedSignal();
		observe.push_back(b);
		observe.push_back(!b);
	}

	_dm->setObserve(observe);
}

vector<bool> Snapshot::generateSignal(const vector<string> &mids) {
	vector<bool> results;
	for (int i = 0; i < _sensors.size(); ++i) {
		if (std::find(mids.begin(), mids.end(), _sensors[i]->_m->_uuid) != mids.end())
			results.push_back(true);
		else
			results.push_back(false);
		if (std::find(mids.begin(), mids.end(), _sensors[i]->_cm->_uuid) != mids.end())
			results.push_back(true);
		else
			results.push_back(false);
	}
	return results;
}

vector<bool> Snapshot::generateSignal(const vector<AttrSensor*> &m) {
	vector<string> mids;
	for (int i = 0; i < m.size(); ++i) {
		mids.push_back(m[i]->_uuid);
	}
	
	return generateSignal(mids);
}

void Snapshot::updateTotal(double phi, bool active) {
	_total_ = _total;
	if (active) {
		_total = _q * _total + (1 - _q) * phi;
	}
}

void Snapshot::updateQ() {
	_q=stod(_ppm->get("q"));
	snapshotLogger.debug("Setting q value to " + to_string(_q), this->getParentChain());
}


void Snapshot::saveSnapshot(ofstream &file) {
	//write uuid
	int uuidLength = _uuid.length();
	file.write(reinterpret_cast<const char *>(&uuidLength), sizeof(int));
	file.write(_uuid.c_str(), uuidLength * sizeof(char));
	file.write(reinterpret_cast<const char *>(&_type), sizeof(UMA_SNAPSHOT));
	file.write(reinterpret_cast<const char *>(&_initialSize), sizeof(int));

	int sensorSize = _sensors.size();
	int sensorPairSize = _sensorPairs.size();
	file.write(reinterpret_cast<const char *>(&sensorSize), sizeof(int));
	file.write(reinterpret_cast<const char *>(&sensorPairSize), sizeof(int));

	_dm->copyArraysToSensors(0, sensorSize, _sensors);
	_dm->copyArraysToSensorPairs(0, sensorSize, _sensorPairs);
	for (int i = 0; i < sensorSize; ++i) {
		_sensors[i]->saveSensor(file);
	}
	for (int i = 0; i < sensorPairSize; ++i) {
		_sensorPairs[i]->saveSensorPair(file);
	}
}

Snapshot *Snapshot::loadSnapshot(ifstream &file, UMACoreObject *parent) {
	int uuidLength = -1;
	file.read((char *)(&uuidLength), sizeof(int));

	string uuid = string(uuidLength, ' ');
	file.read(&uuid[0], uuidLength * sizeof(char));
	
	UMA_SNAPSHOT type = UMA_SNAPSHOT::SNAPSHOT_STATIONARY;
	file.read((char *)(&type), sizeof(UMA_SNAPSHOT));

	int initialSize = -1;
	file.read((char *)(&initialSize), sizeof(int));

	Snapshot *snapshot = nullptr;
	switch (type) {
	case UMA_SNAPSHOT::SNAPSHOT_QUALITATIVE: snapshot = new SnapshotQualitative(uuid, parent); break;
	case UMA_SNAPSHOT::SNAPSHOT_DISCOUNTED: snapshot = new SnapshotDiscounted(uuid, parent); break;
	case UMA_SNAPSHOT::SNAPSHOT_EMPIRICAL: snapshot = new SnapshotEmpirical(uuid, parent); break;
	default: snapshot = new Snapshot(uuid, parent);
	}

	int sensorSize = -1;
	int sensorPairSize = -1;
	file.read((char *)(&sensorSize), sizeof(int));
	file.read((char *)(&sensorPairSize), sizeof(int));
	snapshotLogger.debug("will load " + to_string(sensorSize) + " sensors and " + to_string(sensorPairSize) + " sensor pairs", snapshot->getParentChain());

	for (int i = 0; i < sensorSize; ++i) {
		snapshot->_sensors.push_back(Sensor::loadSensor(file, snapshot));
	}
	for (int i = 0; i < sensorPairSize; ++i) {
		snapshot->_sensorPairs.push_back(SensorPair::loadSensorPair(file, snapshot->_sensors, snapshot));
	}

	if (snapshot->_sensors.size() != initialSize) {
		snapshotLogger.error("The initialSize=" + to_string(initialSize) + 
			" is not matching the _sensors.size=" + to_string(snapshot->_sensors.size()), snapshot->getParentChain());
	}
	snapshot->setInitialSize();

	snapshotLogger.info("snapshot is successfully loaded!", snapshot->getParentChain());
	return snapshot;
}

/*
----------------Snapshot Base Class-------------------
*/

/*
----------------SnapshotQualitative Class-------------------
*/

SnapshotQualitative::SnapshotQualitative(const string &uuid, UMACoreObject *parent)
	:Snapshot(uuid, parent, UMA_SNAPSHOT::SNAPSHOT_QUALITATIVE) {
}

SnapshotQualitative::~SnapshotQualitative() {}

void SnapshotQualitative::updateTotal(double phi, bool active) {
	_total_ = _total;
	if (active) {
		_total = (_total < -0.5 || phi < _total) ? phi : _total;
	}
}

void SnapshotQualitative::generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair) {
	//create a new delayed sensor
	Sensor *delayedSensor = new Sensor(idPair, this, _total, _sensors.size());
	_sensorIdx[idPair.first] = delayedSensor;
	_sensorIdx[idPair.second] = delayedSensor;
	vector<SensorPair *> delayedSensorPairs;
	//the sensor name is TBD, need python input
	int sid = mid / 2;
	//get mid and sid

	bool isSensorActive;
	if (merge) {
		//if need to merge, means just a single sensor delay
		isSensorActive = _sensors[sid]->getObserve();
	}
	else {
		//means not a single sensor delay
		_sensors[sid]->setObserveList(_dm->h_observe, _dm->h_observe_);
		isSensorActive = _sensors[sid]->generateDelayedSignal();
	}
	//reverse for compi
	if (mid % 2 == 1) isSensorActive = !isSensorActive;

	for (int i = 0; i < _sensors.size() + 1; ++i) {
		SensorPair *sensorPair = NULL;
		if (i == _sensors.size()) {
			//if this is the last sensor pair, and it is the pair of the delayed sensor itself
			sensorPair = new SensorPair(this, delayedSensor, delayedSensor, _threshold, _total);
			//copy all those diag values first
			delayedSensor->_m->_vdiag = qless(delayedSensorPairs[0]->mij->_vw, delayedSensorPairs[0]->mi_j->_vw) ? delayedSensorPairs[0]->mij->_vw : delayedSensorPairs[0]->mi_j->_vw;
			delayedSensor->_cm->_vdiag = qless(delayedSensorPairs[0]->m_ij->_vw, delayedSensorPairs[0]->m_i_j->_vw) ? delayedSensorPairs[0]->m_ij->_vw : delayedSensorPairs[0]->m_i_j->_vw;
			delayedSensor->_m->_vdiag_ = delayedSensor->_m->_vdiag;
			delayedSensor->_cm->_vdiag_ = delayedSensor->_cm->_vdiag;
			
			sensorPair->mij->_vw = delayedSensor->_m->_vdiag;
			sensorPair->mi_j->_vw = -1;
			sensorPair->m_ij->_vw = -1;
			sensorPair->m_i_j->_vw = delayedSensor->_cm->_vdiag;
			delayedSensorPairs.push_back(sensorPair);
		}
		else {
			sensorPair = new SensorPair(this, delayedSensor, _sensors[i], _threshold, _total);
			sensorPair->mij->_vw = isSensorActive * _sensors[i]->_m->_vdiag + !isSensorActive * -1;
			sensorPair->mi_j->_vw = isSensorActive * _sensors[i]->_cm->_vdiag + !isSensorActive * -1;
			sensorPair->m_ij->_vw = !isSensorActive * _sensors[i]->_m->_vdiag + isSensorActive * -1;
			sensorPair->m_i_j->_vw = !isSensorActive * _sensors[i]->_cm->_vdiag + isSensorActive * -1;
			delayedSensorPairs.push_back(sensorPair);
		}
	}

	if (!merge) {
		//replace the last one
		Sensor *oldSensor = _sensors[sid];
		//set the mask amper of the delayed sensor to be the same as the one that need to be delayed
		delayedSensor->setAmperList(oldSensor);
		//take the idx of the old one, copy it to the new one 
		delayedSensor->setIdx(oldSensor->_idx);
		//delete the old one
		delete oldSensor;
		//if no mrege needed, means the last existing sensor and row of sensorPair need to be removed
		_sensors.pop_back();
		//destruct the last row of sensor pair
		for (int i = ind(_sensors.size(), 0); i < ind(_sensors.size() + 1, 0); ++i) {
			delete _sensorPairs[i];
		}
		_sensorPairs.erase(_sensorPairs.begin() + ind(_sensors.size(), 0), _sensorPairs.end());
		//also need to remove the n-1 position of delayedSensor_pair
		delete delayedSensorPairs[_sensors.size()];
		delayedSensorPairs.erase(delayedSensorPairs.end() - 2, delayedSensorPairs.end() - 1);
	}
	else {
		delayedSensor->setAmperList(mid);
	}
	_sensors.push_back(delayedSensor);
	_sensorPairs.insert(_sensorPairs.end(), delayedSensorPairs.begin(), delayedSensorPairs.end());
}

void SnapshotQualitative::updateQ() {
}

/*
----------------SnapshotQualitative Class-------------------
*/

/*
----------------SnapshotDiscounted Class-------------------
*/

SnapshotDiscounted::SnapshotDiscounted(const string &uuid, UMACoreObject *parent)
	:Snapshot(uuid, parent, UMA_SNAPSHOT::SNAPSHOT_DISCOUNTED) {
}

SnapshotDiscounted::~SnapshotDiscounted() {}

void SnapshotDiscounted::updateTotal(double phi, bool active) {
	Snapshot::updateTotal(phi, active);
}

void SnapshotDiscounted::generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair) {
	Snapshot::generateDelayedWeights(mid, merge, idPair);
}

void SnapshotDiscounted::updateQ() {
	Snapshot::updateQ();
}

/*
----------------SnapshotDiscounted Class-------------------
*/

/*
----------------SnapshotEmpirical Class-------------------
*/

SnapshotEmpirical::SnapshotEmpirical(const string &uuid, UMACoreObject *parent)
	:Snapshot(uuid, parent, UMA_SNAPSHOT::SNAPSHOT_EMPIRICAL) {
}

SnapshotEmpirical::~SnapshotEmpirical() {}

void SnapshotEmpirical::updateTotal(double phi, bool active) {
	Snapshot::updateTotal(phi, active);
}

void SnapshotEmpirical::generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair) {
	Snapshot::generateDelayedWeights(mid, merge, idPair);
}

void SnapshotEmpirical::updateQ() {
	_q = 1.0 / (2.0 - _q);
}

/*
----------------SnapshotDiscounted Class-------------------
*/