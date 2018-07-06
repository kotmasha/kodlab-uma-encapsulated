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

/*
----------------Snapshot Base Class-------------------
*/
extern int ind(int row, int col);
extern int compi(int x);
//extern std::map<string, std::map<string, string>> server_cfg;
static Logger snapshotLogger("Snapshot", "log/snapshot.log");

Snapshot::Snapshot(const string &uuid, const string &dependency, const int type) : _uuid(uuid), _dependency(dependency + ":" + uuid), _type(type) {
	_total = stod(World::coreInfo["Snapshot"]["total"]);
	_total_ = _total;
	snapshotLogger.debug("Setting init total value to " + to_string(_total), _dependency);

	_delayCount = 0;
	
	_q = stod(World::coreInfo["Snapshot"]["q"]);
	snapshotLogger.debug("Setting q value to " + to_string(_q), _dependency);

	_threshold = stod(World::coreInfo["Snapshot"]["threshold"]);
	snapshotLogger.debug("Setting threshold value to " + to_string(_threshold), _dependency);

	_autoTarget = stoi(World::coreInfo["Snapshot"]["auto_target"]);
	snapshotLogger.debug("Setting auto target value to " + to_string(_autoTarget), _dependency);

	_propagateMask = stoi(World::coreInfo["Snapshot"]["propagate_mask"]);
	snapshotLogger.debug("Setting propagate mask value to " + to_string(_propagateMask), _dependency);

	_initialSize = 0;

	_dm = new DataManager(_dependency);
	snapshotLogger.info("Data Manager is created", _dependency);

	snapshotLogger.info("A Snapshot " + _uuid + " is created, with type " + to_string(_type), _dependency);
}


Snapshot::~Snapshot(){
	try {
		for (int i = 0; i < _sensors.size(); ++i) {
			delete _sensors[i];
			_sensors[i] = NULL;
		}
		snapshotLogger.debug("All snapshot sensors deleted", _dependency);
		for (int i = 0; i < _sensorPairs.size(); ++i) {
			delete _sensorPairs[i];
			_sensorPairs[i] = NULL;
		}
		snapshotLogger.debug("All snapshot sensor pairs deleted", _dependency);
		delete _dm;
		snapshotLogger.debug("Data Manager deleted", _dependency);
	}
	catch (exception &e) {
		snapshotLogger.error("Fatal error when trying to delete snapshot: " + _uuid, _dependency);
		throw UMAException("Fatal error in Snapshot destruction function", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
	}
	snapshotLogger.info("Deleted the snapshot: " + _uuid, _dependency);
}

Sensor *Snapshot::createSensor(const std::pair<string, string> &id_pair, const vector<double> &diag, const vector<vector<double> > &w, const vector<vector<bool> > &b) {
	_dm->copyArraysToSensors(0, _sensors.size(), _sensors);
	_dm->copyArraysToSensorPairs(0, _sensors.size(), _sensorPairs);
	if (_sensorIdx.find(id_pair.first) != _sensorIdx.end() && _sensorIdx.find(id_pair.second) != _sensorIdx.end()) {
		snapshotLogger.error("Cannot create a duplicate sensor!", _dependency);
		throw UMAException("Cannot create a duplicate sensor!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::DUPLICATE);
	}
	Sensor *sensor = NULL;
	if (diag.empty()) {
		sensor = new Sensor(id_pair, _total, _sensors.size());
	}
	else {
		sensor = new Sensor(id_pair, diag, _sensors.size());
	}
	_sensorIdx[id_pair.first] = sensor;
	_sensorIdx[id_pair.second] = sensor;
	_sensors.push_back(sensor);
	snapshotLogger.debug("A Sensor " + id_pair.first + " is created with idx " + to_string(sensor->_idx), _dependency);
	//creating sensor pairs
	for (int i = 0; i < _sensors.size(); ++i) {
		SensorPair *sensor_pair = NULL;
		if (w.empty()) {
			sensor_pair = new SensorPair(sensor, _sensors[i], _threshold, _total);
		}
		else {
			sensor_pair = new SensorPair(sensor, _sensors[i], _threshold, w[i], b[i]);
		}
		snapshotLogger.debug("A sensor pair with sensor1=" + sensor->_uuid + " sensor2=" + _sensors[i]->_uuid + " is created", _dependency);
		_sensorPairs.push_back(sensor_pair);
	}
	snapshotLogger.info(to_string(_sensors.size()) + " Sensor Pairs are created, total is " + to_string(ind(_sensors.size(), 0)), _dependency);
	
	if (_sensors.size() > _dm->_sensorSizeMax) {
		snapshotLogger.debug("Need allocate more space after adding a sensor", _dependency);
		_dm->reallocateMemory(_total, _sensors.size());
		_dm->createSensorsToArraysIndex(0, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(0, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(0, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(0, _sensors.size(), _sensorPairs);
	}
	else {
		snapshotLogger.debug("Have enough space, will not do remalloc", _dependency);
		_dm->setSize(_sensors.size(), false);
		_dm->createSensorsToArraysIndex(_sensors.size() - 1, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(_sensors.size() - 1, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(_sensors.size() - 1, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(_sensors.size() - 1, _sensors.size(), _sensorPairs);
	}
	return sensor;
}

void Snapshot::deleteSensor(const string &sensor_id) {
	if (_sensorIdx.find(sensor_id) == _sensorIdx.end()) {
		snapshotLogger.error("Cannot find the sensor " + sensor_id, _dependency);
		throw UMAException("Cannot find the sensor " + sensor_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}

	int sensor_idx = _sensorIdx.at(sensor_id)->_idx;
	vector<bool> pruning_list(_dm->_attrSensorSize, false);
	pruning_list[2 * sensor_idx] = true;
	pruning(pruning_list);
	snapshotLogger.info("Sensor " + sensor_id + " deleted", _dependency);
}

vector<vector<string> > Snapshot::getSensorInfo() const {
	vector<vector<string>> results;
	for (int i = 0; i < _sensors.size(); ++i) {
		Sensor * const sensor = _sensors[i];
		vector<string> sensor_pair = { _sensors[i]->_m->_uuid, _sensors[i]->_cm->_uuid };
		results.push_back(sensor_pair);
	}
	return results;
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
		throw UMAException("Input signal size for pruning is larger than attr_sensorSize", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	//get converted sensor list, from attr_sensor signal
	const vector<bool> sensor_list = SignalUtil::attrSensorToSensorSignal(signal);
	const vector<int> idx_list = SignalUtil::boolSignalToIntIdx(sensor_list);
	if (idx_list.empty()) {
		snapshotLogger.info("Empty pruning signal, do nothing", _dependency);
		return;
	}
	_dm->copyArraysToSensors(0, _sensors.size(), _sensors);
	_dm->copyArraysToSensorPairs(0, _sensors.size(), _sensorPairs);

	if (idx_list[0] < 0 || idx_list.back() >= _sensors.size()) {
		throw UMAException("Pruning range is from " + to_string(idx_list[0]) + "~" + to_string(idx_list.back()) + ", illegal range!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}

	string str_list = "";
	for (int i = 0; i < idx_list.size(); ++i) {
		str_list += to_string(idx_list[i]) + ", ";
	}
	snapshotLogger.info("Will prune id=" + str_list, _dependency);

	int row_escape = 0;
	int total_escape = 0;
	//destruct the corresponding sensors and sensor pairs
	for(int i = 0; i < _sensors.size(); ++i){
		if(sensor_list[i]){
			//delete the sensor if necessary
			_sensorIdx.erase(_sensors[i]->_m->_uuid);
			_sensorIdx.erase(_sensors[i]->_cm->_uuid);
			vector<int> amper_list = _sensors[i]->getAmperList();
			vector<bool> amper_signal = SignalUtil::intIdxToBoolSignal(amper_list, _sensors.size() * 2);
			size_t delay_list_hash = delayHash(amper_signal);
			_delaySensorHash.erase(delay_list_hash);

			delete _sensors[i];
			_sensors[i] = NULL;
			row_escape++;
		}
		else{
			//or just adjust the idx of the sensor, and change the position
			vector<int> amper_list = _sensors[i]->getAmperList();
			vector<int> new_amper_list;
			for (int j = 0; j < amper_list.size(); ++j) {
				int idx = ArrayUtil::findIdxInSortedArray(idx_list, amper_list[j] / 2);
				if (idx >= 0 && idx_list[idx] != amper_list[j] / 2) {//if the value is not to be pruned
					new_amper_list.push_back(amper_list[j] - 2 * (idx + 1));
				}
			}

			size_t old_delayHash = delayHash(SignalUtil::intIdxToBoolSignal(amper_list, _sensors.size() * 2));
			size_t new_delayHash = delayHash(SignalUtil::intIdxToBoolSignal(new_amper_list, _sensors.size() * 2));
			_delaySensorHash.erase(old_delayHash);
			_delaySensorHash.insert(new_delayHash);

			_sensors[i]->_amper.clear();
			_sensors[i]->_amper = new_amper_list;
			_sensors[i]->setIdx(i - row_escape);
			_sensors[i - row_escape] = _sensors[i];
		}
		//delete the row of sensor, or the col in a row, where ther other sensor is deleted
		for(int j = 0; j <= i; ++j){
			if(sensor_list[i] || sensor_list[j]){
				//delete the sensor pair if necessary
				delete _sensorPairs[ind(i, j)];
				_sensorPairs[ind(i, j)] = NULL;
				total_escape++;
			}
			else{
				//or just change the position
				_sensorPairs[ind(i, j) - total_escape] = _sensorPairs[ind(i, j)];
			}
		}
	}

	//earse the additional space
	_sensors.erase(_sensors.end() - row_escape, _sensors.end());
	_sensorPairs.erase(_sensorPairs.end() - total_escape, _sensorPairs.end());
	//adjust the size variables
	_dm->setSize(_sensors.size(), false);

	_dm->createSensorsToArraysIndex(0, _sensors.size(), _sensors);
	_dm->createSensorPairsToArraysIndex(0, _sensors.size(), _sensorPairs);
	_dm->copySensorsToArrays(0, _sensors.size(), _sensors);
	_dm->copySensorPairsToArrays(0, _sensors.size(), _sensorPairs);

	snapshotLogger.info("Pruning done successful", _dependency);
}

void Snapshot::ampers(const vector<vector<bool> > &lists, const vector<std::pair<string, string> > &id_pairs){
	_dm->copyArraysToSensors(0, _sensors.size(), _sensors);
	_dm->copyArraysToSensorPairs(0, _sensors.size(), _sensorPairs);
	int success_amper = 0;
	//record how many delay are successful
	
	for(int i = 0; i < lists.size(); ++i){
		const vector<int> list = SignalUtil::boolSignalToIntIdx(lists[i]);
		if (list.size() < 2) {
			snapshotLogger.warn("The amper vector size is less than 2, will abort this amper operation, list id: " + to_string(i), _dependency);
			continue;
		}
		amper(list, id_pairs[i]);
		success_amper++;
	}

	snapshotLogger.info(to_string(success_amper) + " out of " + to_string(lists.size()) + " amper successfully done", _dependency);

	if(_sensors.size() > _dm->_sensorSizeMax){
		snapshotLogger.info("New sensor size larger than current max, will resize", _dependency);
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
		_dm->createSensorsToArraysIndex(_sensors.size() - success_amper, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(_sensors.size() - success_amper, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(_sensors.size() - success_amper, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(_sensors.size() - success_amper, _sensors.size(), _sensorPairs);
	}
}


void Snapshot::delays(const vector<vector<bool> > &lists, const vector<std::pair<string, string> > &id_pairs) {
	_dm->copyArraysToSensors(0, _sensors.size(), _sensors);
	_dm->copyArraysToSensorPairs(0, _sensors.size(), _sensorPairs);

	bool generate_default_id = id_pairs.empty();
	pair<string, string> p;
	int success_delay = 0;
	//record how many delay are successful
	for (int i = 0; i < lists.size(); ++i) {
		size_t v = delayHash(lists[i]);
		if (_delaySensorHash.end() != _delaySensorHash.find(v)) {
			snapshotLogger.info("Find an existing delayed sensor, will skip creating current one", _dependency);
			continue;
		}

		if (lists[i].size() > _sensors.size() * 2) {
			throw UMAException("The " + to_string(i) + "th input signal size is larger than 2 * sensors.size()", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
		}
		const vector<int> list = SignalUtil::boolSignalToIntIdx(lists[i]);
		if (list.size() < 1) {
			snapshotLogger.warn("The amper vector size is less than 1, will abort this amper operation, list id: " + to_string(i), _dependency);
			continue;
		}
		if(generate_default_id){
			int delay_count = getDealyCount();
			p = { "delay" + to_string(delay_count), "c_delay" + to_string(delay_count) };
		}
		else {
			p = id_pairs[i];
		}
		
		if (list.size() == 1) {
			try {
				generateDelayedWeights(list[0], true, p);
			}
			catch (UMAException &e) {
				throw UMAException("Fatal error in generateDelayedWeights", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
			}
		}
		else {
			amper(list, p);
			try {
				generateDelayedWeights(_sensors.back()->_m->_idx, false, p);
			}
			catch (UMAException &e) {
				throw UMAException("Fatal error in generateDelayedWeights", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
			}
		}
		success_delay++;
		_delaySensorHash.insert(v);

		snapshotLogger.info("A delayed sensor is generated " + p.first, _dependency);
		string delay_list = "";
		for (int j = 0; j < list.size(); ++j) delay_list += (to_string(list[j]) + ",");
		snapshotLogger.verbose("The delayed sensor generated from " + delay_list, _dependency);
	}

	snapshotLogger.info(to_string(success_delay) + " out of " + to_string(lists.size()) + " delay successfully done", _dependency);

	if (_sensors.size() > _dm->_sensorSizeMax) {
		snapshotLogger.info("New sensor size larger than current max, will resize", _dependency);
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
		_dm->createSensorsToArraysIndex(_sensors.size() - success_delay, _sensors.size(), _sensors);
		_dm->createSensorPairsToArraysIndex(_sensors.size() - success_delay, _sensors.size(), _sensorPairs);
		_dm->copySensorsToArrays(_sensors.size() - success_delay, _sensors.size(), _sensors);
		_dm->copySensorPairsToArrays(_sensors.size() - success_delay, _sensors.size(), _sensorPairs);
	}
}

//####################################################################################
//------------------------------------SET FUNCTION------------------------------------

void Snapshot::setThreshold(const double &threshold) {
	_threshold = threshold;
	snapshotLogger.info("snapshot threshold changed to " + to_string(threshold), _dependency);
}

void Snapshot::setQ(const double &q) {
	_q = q;
	snapshotLogger.info("snapshot q changed to " + to_string(q), _dependency);
}

void Snapshot::setAutoTarget(const bool &auto_target) {
	_autoTarget = auto_target;
}

void Snapshot::setPropagateMask(const bool &propagateMask) {
	_propagateMask = propagateMask;
}

void Snapshot::setInitialSize(const int &initial_size) {
	_initialSize = initial_size;
}

void Snapshot::setInitialSize() {
	_initialSize = _sensors.size();
}

void Snapshot::setTotal(const double &total){
	_total = total;
}

void Snapshot::setOldTotal(const double &total_) {
	_total_ = total_;
}


//------------------------------------SET FUNCTION------------------------------------
//####################################################################################

//####################################################################################
//------------------------------------GET FUNCTION------------------------------------

/*
this function is getting the attr_sensor, from the sensor list
*/
AttrSensor *Snapshot::getAttrSensor(int idx) const{
	int s_idx = idx / 2;
	if (s_idx >= _sensors.size() || s_idx <0) {
		throw UMAException("the input attr_sensor index is out of range, input is " + to_string(s_idx) + " sensor num is " + to_string(idx), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	if(idx % 2 == 0){
		return _sensors[s_idx]->_m;
	}
	else{
		return _sensors[s_idx]->_cm;
	}
}

AttrSensor *Snapshot::getAttrSensor(const string &attr_sensor_id) const{
	Sensor *sensor = getSensor(attr_sensor_id);
	if (attr_sensor_id == sensor->_m->_uuid) return sensor->_m;
	else if(attr_sensor_id == sensor->_cm->_uuid) return sensor->_cm;
	throw UMAException("Cannot find the attr_sensor id " + attr_sensor_id, UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
}

SensorPair *Snapshot::getSensorPair(const Sensor *sensor1, const Sensor *sensor2) const{
	int idx1 = sensor1->_idx > sensor2->_idx ? sensor1->_idx : sensor2->_idx;
	int idx2 = sensor1->_idx > sensor2->_idx ? sensor2->_idx : sensor1->_idx;
	return _sensorPairs[ind(idx1, idx2)];
}

/*
This function is getting the attr_sensor pair from the sensor pair list
Input: m_idx1, m_idx2 are index of the attr_sensor, m_idx1 > m_idx2
*/
AttrSensorPair *Snapshot::getAttrSensorPair(int m_idx1, int m_idx2) const{
	int idx1 = m_idx1 > m_idx2 ? m_idx1 : m_idx2;
	int idx2 = m_idx1 > m_idx2 ? m_idx2 : m_idx1;
	int s_idx1 = idx1 / 2;
	int s_idx2 = idx2 / 2;
	AttrSensor *m1 = getAttrSensor(idx1);
	AttrSensor *m2 = getAttrSensor(idx2);
	return _sensorPairs[ind(s_idx1, s_idx2)]->getAttrSensorPair(m1->_isOriginPure, m2->_isOriginPure);
}

AttrSensorPair *Snapshot::getAttrSensorPair(const string &mid1, const string &mid2) const{
	int idx1 = getAttrSensor(mid1)->_idx;
	int idx2 = getAttrSensor(mid2)->_idx;
	return getAttrSensorPair(idx1, idx2);
}

vector<bool> Snapshot::getAmperList(const string &sensor_id) const{
	if (_sensorIdx.find(sensor_id) == _sensorIdx.end()) {
		throw UMAException("Cannot find the sensor id " + sensor_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
	}
	Sensor *sensor = _sensorIdx.at(sensor_id);
	vector<bool> result(_dm->_attrSensorSize, false);
	for (int i = 0; i < sensor->_amper.size(); ++i) {
		result[sensor->_amper[i]] = true;
	}
	return result;
}

vector<string> Snapshot::getAmperListID(const string &sensor_id) const{
	if (_sensorIdx.find(sensor_id) == _sensorIdx.end()) {
		throw UMAException("Cannot find the sensor id " + sensor_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
	}
	Sensor * const sensor = _sensorIdx.at(sensor_id);
	vector<string> result;
	for (int i = 0; i < sensor->_amper.size(); ++i) {
		int idx = sensor->_amper[i];
		Sensor *s = _sensors[idx / 2];
		if (idx % 2 == 0) result.push_back(s->_m->_uuid);
		else result.push_back(s->_cm->_uuid);
	}
	return result;
}

Sensor *Snapshot::getSensor(const string &sensor_id) const{
	if (_sensorIdx.find(sensor_id) == _sensorIdx.end()) {
		throw UMAException("Cannot find the sensor id " + sensor_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
	}
	return _sensorIdx.at(sensor_id);
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

const int &Snapshot::getType() const {
	return _type;
}

const int Snapshot::getDealyCount() {
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
		snapshotLogger.warn("Amper list size is smaller than 2, will not continue", _dependency);
		return;
	}
	try {
		amperand(list[1], list[0], true, uuid);
		for (int j = 2; j < list.size(); ++j) {
			amperand(_sensors.back()->_m->_idx, list[j], false, uuid);
		}
	}
	catch (UMAException &e) {
		throw UMAException("Fatal error while doing amper and", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
	}
}

/*
This is the amper and function, used in amper and delay
Input: m_idx1, m_idx2 are the attr_sensor idx that need to be amperand, m_idx1 > m_idx2, merge is indicating whether merge or replace the last sensor/row of sensor pair
*/
void Snapshot::amperand(int m_idx1, int m_idx2, bool merge, const std::pair<string, string> &id_pair) {
	vector<SensorPair*> amper_and_sensorPairs;
	Sensor *amper_and_sensor = new Sensor(id_pair, _total, _sensors.size());
	_sensorIdx[id_pair.first] = amper_and_sensor;
	_sensorIdx[id_pair.second] = amper_and_sensor;

	double f = 1.0;
	if(_total > 1e-5)
		f = getAttrSensorPair(m_idx1, m_idx2)->_vw / _total;
	for(int i = 0; i < _sensors.size(); ++i){
		SensorPair *sensor_pair = NULL;
		sensor_pair = new SensorPair(amper_and_sensor, _sensors[i], _threshold, _total);
		AttrSensor *m1 = _sensors[i]->_m;
		AttrSensor *m2 = _sensors[i]->_cm;

		if (_sensors[i]->_m->_idx == m_idx1 || _sensors[i]->_m->_idx == m_idx2) {
			sensor_pair->mij->_vw = getAttrSensorPair(m_idx1, m_idx2)->_vw;
			sensor_pair->mi_j->_vw = 0.0;
			if (_sensors[i]->_m->_idx == m_idx1) {
				sensor_pair->m_ij->_vw = getAttrSensorPair(m_idx1, compi(m_idx2))->_vw;
				sensor_pair->m_i_j->_vw = getAttrSensorPair(compi(m_idx1), compi(m_idx1))->_vw;
			}
			else {
				sensor_pair->m_ij->_vw = getAttrSensorPair(compi(m_idx1), m_idx2)->_vw;
				sensor_pair->m_i_j->_vw = getAttrSensorPair(compi(m_idx2), compi(m_idx2))->_vw;
			}
		}
		else if (_sensors[i]->_cm->_idx == m_idx1 || _sensors[i]->_cm->_idx == m_idx2) {
			sensor_pair->mi_j->_vw = getAttrSensorPair(m_idx1, m_idx2)->_vw;
			sensor_pair->mij->_vw = 0.0;
			if (_sensors[i]->_cm->_idx == m_idx1) {
				sensor_pair->m_i_j->_vw = getAttrSensorPair(m_idx1, compi(m_idx2))->_vw;
				sensor_pair->m_ij->_vw = getAttrSensorPair(compi(m_idx1), compi(m_idx1))->_vw;
			}
			else {
				sensor_pair->m_i_j->_vw = getAttrSensorPair(compi(m_idx1), m_idx2)->_vw;
				sensor_pair->m_ij->_vw = getAttrSensorPair(compi(m_idx2), compi(m_idx2))->_vw;
			}
		}
		else {
			sensor_pair->mij->_vw = f * getAttrSensorPair(m1->_idx, m1->_idx)->_vw;
			sensor_pair->mi_j->_vw = f * getAttrSensorPair(m2->_idx, m2->_idx)->_vw;
			sensor_pair->m_ij->_vw = (1.0 - f) * getAttrSensorPair(m1->_idx, m1->_idx)->_vw;
			sensor_pair->m_i_j->_vw = (1.0 - f) * getAttrSensorPair(m2->_idx, m2->_idx)->_vw;
		}
		amper_and_sensorPairs.push_back(sensor_pair);
	}
	SensorPair *self_pair = new SensorPair(amper_and_sensor, amper_and_sensor, _threshold, _total);
	self_pair->mij->_vw = amper_and_sensorPairs[0]->mij->_vw + amper_and_sensorPairs[0]->mi_j->_vw;
	self_pair->mi_j->_vw = 0.0;
	self_pair->m_ij->_vw = 0.0;
	self_pair->m_i_j->_vw = amper_and_sensorPairs[0]->m_ij->_vw + amper_and_sensorPairs[0]->m_i_j->_vw;
	amper_and_sensorPairs.push_back(self_pair);
	if(!merge){
		Sensor *old_sensor = _sensors.back();
		//restore the old sensor amper list
		amper_and_sensor->setAmperList(old_sensor);
		//also append the m_idx2 as the new amper list value
		amper_and_sensor->setAmperList(m_idx2);
		//take the idx of the old one, copy it to the new one 
		amper_and_sensor->setIdx(old_sensor->_idx);
		//delete the old one
		delete old_sensor;
		//if no mrege needed, means the last existing sensor and row of sensor_pair need to be removed
		_sensors.pop_back();
		//destruct the last row of sensor pair
		for(int i = ind(_sensors.size(), 0); i < ind(_sensors.size() + 1, 0); ++i){
			delete _sensorPairs[i];
		}
		_sensorPairs.erase(_sensorPairs.begin() + ind(_sensors.size(), 0), _sensorPairs.end());
		//also need to remove the n-1 position of amper_and_sensorPairs
		delete amper_and_sensorPairs[_sensors.size()];
		amper_and_sensorPairs.erase(amper_and_sensorPairs.end() - 2, amper_and_sensorPairs.end() - 1);
	}
	else{
		//set the amper list, if append a new sensor, append the smaller idx first
		amper_and_sensor->setAmperList(m_idx2);
		amper_and_sensor->setAmperList(m_idx1);
	}
	_sensors.push_back(amper_and_sensor);
	_sensorPairs.insert(_sensorPairs.end(), amper_and_sensorPairs.begin(), amper_and_sensorPairs.end());
}

/*
This function is generating the delayed weights
Before this function is called, have to make sure, _sensors and _sensorPairs have valid info in vwxx;
Input: mid of the attr_sensor doing the delay, and whether to merge after the operation
*/
void Snapshot::generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &id_pair){
	//create a new delayed sensor
	Sensor *delayedSensor = new Sensor(id_pair, _total, _sensors.size());
	_sensorIdx[id_pair.first] = delayedSensor;
	_sensorIdx[id_pair.second] = delayedSensor;
	vector<SensorPair *> delayedSensorPairs;
	//the sensor name is TBD, need python input
	int sid = mid / 2;
	//get mid and sid

	bool is_sensor_active;
	if(merge){
		//if need to merge, means just a single sensor delay
		is_sensor_active = _sensors[sid]->getObserve();
	}
	else{
		//means not a single sensor delay
		_sensors[sid]->setObserveList(_dm->h_observe, _dm->h_observe_);
		is_sensor_active = _sensors[sid]->generateDelayedSignal();
	}
	//reverse for compi
	if (mid % 2 == 1) is_sensor_active = !is_sensor_active;

	for(int i = 0; i < _sensors.size() + 1; ++i){
		SensorPair *sensor_pair = NULL;
		if(i == _sensors.size()){
			//if this is the last sensor pair, and it is the pair of the delayed sensor itself
			sensor_pair = new SensorPair(delayedSensor, delayedSensor, _threshold, _total);
			//copy all those diag values first
			delayedSensor->_m->_vdiag = delayedSensorPairs[0]->mij->_vw + delayedSensorPairs[0]->mi_j->_vw;
			delayedSensor->_cm->_vdiag = delayedSensorPairs[0]->m_ij->_vw + delayedSensorPairs[0]->m_i_j->_vw;
			delayedSensor->_m->_vdiag_ = delayedSensor->_m->_vdiag;
			delayedSensor->_cm->_vdiag_ = delayedSensor->_cm->_vdiag;
			//then assign the value to sensor pair
			sensor_pair->mij->_vw = delayedSensor->_m->_vdiag;
			sensor_pair->mi_j->_vw = 0;
			sensor_pair->m_ij->_vw = 0;
			sensor_pair->m_i_j->_vw = delayedSensor->_cm->_vdiag;
			delayedSensorPairs.push_back(sensor_pair);
		}
		else{
			sensor_pair = new SensorPair(delayedSensor, _sensors[i], _threshold, _total);
			sensor_pair->mij->_vw = is_sensor_active * _sensors[i]->_m->_vdiag;
			sensor_pair->mi_j->_vw = is_sensor_active * _sensors[i]->_cm->_vdiag;
			sensor_pair->m_ij->_vw = !is_sensor_active * _sensors[i]->_m->_vdiag;
			sensor_pair->m_i_j->_vw = !is_sensor_active * _sensors[i]->_cm->_vdiag;
			delayedSensorPairs.push_back(sensor_pair);
		}
	}

	if(!merge){
		//replace the last one
		Sensor *old_sensor = _sensors[sid];
		//set the mask amper of the delayed sensor to be the same as the one that need to be delayed
		delayedSensor->setAmperList(old_sensor);
		//take the idx of the old one, copy it to the new one 
		delayedSensor->setIdx(old_sensor->_idx);
		//delete the old one
		delete old_sensor;
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
		throw UMAException("The input observe signal size is not the 2x initial sensor size", UMAException::ERROR_LEVEL::ERROR, UMAException::CLIENT_DATA);
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

/*
void Snapshot::save_snapshot(ofstream &file) {
	//write uuid
	int uuid_size = _uuid.length();
	file.write(reinterpret_cast<const char *>(&uuid_size), sizeof(int));
	file.write(_uuid.c_str(), uuid_size * sizeof(char));
	int sensor_size = _sensors.size();
	int sensor_pair_size = _sensorPairs.size();
	file.write(reinterpret_cast<const char *>(&sensor_size), sizeof(int));
	file.write(reinterpret_cast<const char *>(&sensor_pair_size), sizeof(int));
	copyArraysToSensors(0, sensor_size);
	copyArraysToSensorPairs(0, sensor_size);
	for (int i = 0; i < sensor_size; ++i) {
		_sensors[i]->save_sensor(file);
	}
	for (int i = 0; i < sensor_pair_size; ++i) {
		_sensorPairs[i]->save_sensor_pair(file);
	}
}
*/
/*
----------------Snapshot Base Class-------------------
*/

/*
----------------SnapshotQualitative Class-------------------
*/
//Snapshot_Stationary::Snapshot_Stationary(ifstream &file, string &log_dir):Snapshot(file, log_dir) {}

SnapshotQualitative::SnapshotQualitative(string uuid, string dependency)
	:Snapshot(uuid, dependency, AGENT_TYPE::QUALITATIVE) {
}

SnapshotQualitative::~SnapshotQualitative() {}

void SnapshotQualitative::updateTotal(double phi, bool active) {
	_total_ = _total;
	if (active) {
		_total = _total > phi ? phi : _total;
	}
}

void SnapshotQualitative::generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &id_pair) {
	//create a new delayed sensor
	Sensor *delayedSensor = new Sensor(id_pair, _total, _sensors.size());
	_sensorIdx[id_pair.first] = delayedSensor;
	_sensorIdx[id_pair.second] = delayedSensor;
	vector<SensorPair *> delayedSensorPairs;
	//the sensor name is TBD, need python input
	int sid = mid / 2;
	//get mid and sid

	for (int i = 0; i < _sensors.size() + 1; ++i) {
		SensorPair *sensor_pair = NULL;
		if (i == _sensors.size()) {
			//if this is the last sensor pair, and it is the pair of the delayed sensor itself
			sensor_pair = new SensorPair(delayedSensor, delayedSensor, _threshold, _total);
			//copy all those diag values first
			delayedSensor->_m->_vdiag = -1;
			delayedSensor->_cm->_vdiag = -1;
			delayedSensor->_m->_vdiag_ = -1;
			delayedSensor->_cm->_vdiag_ = -1;
			//then assign the value to sensor pair
			sensor_pair->mij->_vw = -1;
			sensor_pair->mi_j->_vw = -1;
			sensor_pair->m_ij->_vw = -1;
			sensor_pair->m_i_j->_vw = -1;
			delayedSensorPairs.push_back(sensor_pair);
		}
		else {
			sensor_pair = new SensorPair(delayedSensor, _sensors[i], _threshold, _total);
			sensor_pair->mij->_vw = -1;
			sensor_pair->mi_j->_vw = -1;
			sensor_pair->m_ij->_vw = -1;
			sensor_pair->m_i_j->_vw = -1;
			delayedSensorPairs.push_back(sensor_pair);
		}
	}

	if (!merge) {
		//replace the last one
		Sensor *old_sensor = _sensors[sid];
		//set the mask amper of the delayed sensor to be the same as the one that need to be delayed
		delayedSensor->setAmperList(old_sensor);
		//take the idx of the old one, copy it to the new one 
		delayedSensor->setIdx(old_sensor->_idx);
		//delete the old one
		delete old_sensor;
		//if no mrege needed, means the last existing sensor and row of sensor_pair need to be removed
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

/*
----------------SnapshotQualitative Class-------------------
*/

