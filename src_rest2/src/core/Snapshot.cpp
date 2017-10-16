#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "Measurable.h"
#include "MeasurablePair.h"
#include "logManager.h"
#include "DataManager.h"
#include "UMAException.h"
/*
----------------Snapshot Base Class-------------------
*/
extern int ind(int row, int col);
extern int compi(int x);

/*
Snapshot::Snapshot(ifstream &file, string &log_dir) {
	//write uuid
	int uuid_length = -1;
	file.read((char *)(&uuid_length), sizeof(int));
	_uuid = string(uuid_length, ' ');
	file.read(&_uuid[0], uuid_length * sizeof(char));
	_log_dir = log_dir + _uuid;
	_log = new logManager(logging::VERBOSE, _log_dir, _uuid + ".txt", typeid(*this).name());
	_memory_expansion = .5;

	_log->info() << "Default memory expansion rate is " + to_string(_memory_expansion);

	//set the init value for 0.5 for now, ideally should read such value from conf file
	//TBD: read conf file for _memory_expansion
	_sensor_num = 0;
	_total = 1.0;
	_total_ = 1.0;
	_q = 0.9;
	_threshold = 0.125;
	_auto_target = false;
	_propagate_mask = false;
	_log->debug() << "Setting init total value to " + to_string(_total);

	init_pointers();

	int sensor_size = -1;
	int sensor_pair_size = -1;
	file.read((char *)(&sensor_size), sizeof(int));
	file.read((char *)(&sensor_pair_size), sizeof(int));
	_sensor_num = sensor_size;
	for (int i = 0; i < sensor_size; ++i) {
		Sensor *sensor = new Sensor(file);
		_sensor_idx[sensor->_m->_uuid] = sensor;
		_sensor_idx[sensor->_cm->_uuid] = sensor;
		_sensors.push_back(sensor);
	}
	for (int i = 0; i < sensor_pair_size; ++i) {
		SensorPair *sensor_pair = new SensorPair(file, _sensors);
		_sensor_pairs.push_back(sensor_pair);
	}

	_log->info() << "A Snapshot " + _uuid + " is loaded";
}
*/

Snapshot::Snapshot(string uuid, string log_dir){
	_uuid = uuid;
	_log_dir = log_dir;
	_log = new logManager(logging::VERBOSE, log_dir, uuid + ".txt", typeid(*this).name());

	//set the init value for 0.5 for now, ideally should read such value from conf file
	_total = 1.0;
	_total_ = 1.0;
	_q = 0.9;
	_threshold = 0.125;
	_auto_target = false;
	_propagate_mask = false;
	_initial_size = 0;
	_log->debug() << "Setting init total value to " + to_string(_total);

	_dm = new DataManager(_log_dir);
	_log->info() << "Data Manager is created";

	_log->info() << "A Snapshot " + _uuid + " is created";
}


Snapshot::~Snapshot(){
	try {
		for (int i = 0; i < _sensors.size(); ++i) {
			delete _sensors[i];
			_sensors[i] = NULL;
		}
		for (int i = 0; i < _sensor_pairs.size(); ++i) {
			delete _sensor_pairs[i];
			_sensor_pairs[i] = NULL;
		}
	}
	catch (exception &e) {
		_log->error() << "Fatal error when trying to delete snapshot: " + _uuid;
		throw CoreException("Fatal error in Snapshot destruction function", CoreException::FATAL, status_codes::ServiceUnavailable);
	}
	_log->info() << "Deleted the snapshot: " + _uuid;
}

void Snapshot::update_total(double phi, bool active) {
	_total_ = _total;
	_total = _q * _total + (1 - _q) * phi;
}

float Snapshot::decide(vector<bool> &signal, double phi, bool active){//the decide function
	update_total(phi, active);

	_dm->update_state(signal, _q, phi, _total, _total_, active);
	_dm->halucinate(_initial_size);

	if (_propagate_mask) _dm->propagate_mask();
	if (_auto_target) _dm->calculate_target();

	return _dm->divergence();
}


void Snapshot::add_sensor(std::pair<string, string> &id_pair, vector<double> &diag, vector<vector<double> > &w, vector<vector<bool> > &b) {
	if (_sensor_idx.find(id_pair.first) != _sensor_idx.end() && _sensor_idx.find(id_pair.second) != _sensor_idx.end()) {
		_log->error() << "Cannot create a duplicate sensor!";
		throw CoreException("Cannot create a duplicate sensor!", CoreException::ERROR, status_codes::Conflict);
	}
	Sensor *sensor = NULL;
	if (diag.empty()) {
		sensor = new Sensor(id_pair, _total, _sensors.size());
	}
	else {
		sensor = new Sensor(id_pair, diag, _sensors.size());
	}
	_sensor_idx[id_pair.first] = sensor;
	_sensor_idx[id_pair.second] = sensor;
	_sensors.push_back(sensor);
	_log->debug() << "A Sensor " + id_pair.first + " is created with idx " + to_string(sensor->_idx);
	//creating sensor pairs
	for (int i = 0; i < _sensors.size(); ++i) {
		SensorPair *sensor_pair = NULL;
		if (w.empty()) {
			sensor_pair = new SensorPair(sensor, _sensors[i], _threshold, _total);
		}
		else {
			sensor_pair = new SensorPair(sensor, _sensors[i], _threshold, w[i], b[i]);
		}
		_log->debug() << "A sensor pair with sensor1=" + sensor->_uuid + " sensor2=" + _sensors[i]->_uuid + " is created";
		_sensor_pairs.push_back(sensor_pair);
	}
	_log->info() << to_string(_sensors.size()) + " Sensor Pairs are created, total is " + to_string(ind(_sensors.size(), 0));
	
	if (_sensors.size() > _dm->_sensor_size_max) {
		_log->debug() << "Need allocate more space after adding a sensor";
		_dm->reallocate_memory(_sensors.size());
		_dm->create_sensors_to_arrays_index(0, _sensors.size(), _sensors);
		_dm->create_sensor_pairs_to_arrays_index(0, _sensors.size(), _sensor_pairs);
		_dm->copy_sensors_to_arrays(0, _sensors.size(), _sensors);
		_dm->copy_sensor_pairs_to_arrays(0, _sensors.size(), _sensor_pairs);
	}
	else {
		_log->debug() << "Have enough space, will not do remalloc";
		_dm->set_size(_sensors.size(), false);
		_dm->create_sensors_to_arrays_index(_sensors.size() - 1, _sensors.size(), _sensors);
		_dm->create_sensor_pairs_to_arrays_index(_sensors.size() - 1, _sensors.size(), _sensor_pairs);
		_dm->copy_sensors_to_arrays(_sensors.size() - 1, _sensors.size(), _sensors);
		_dm->copy_sensor_pairs_to_arrays(_sensors.size() - 1, _sensors.size(), _sensor_pairs);
	}
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
		if (snapshot->_sensor_idx.find(sensor_id) != snapshot->_sensor_idx.end()) {
			//if the current sensor id also in the other snapshot

			//copy sensors
			//get the 'same' sensor in the other snapshot
			Sensor *c_sensor = snapshot->_sensor_idx[sensor_id];
			//copy the value
			sensor->copy_data(c_sensor);
			//reconstruct the amper list
			vector<int> amper_list;
			for (int i = 0; i < c_sensor->_amper.size(); ++i) {
				int idx = c_sensor->_amper[i];
				bool isOriginPure = (idx % 2 == 0);
				Sensor *a_sensor = _sensors[idx / 2];
				if (_sensor_idx.find(a_sensor->_uuid) != _sensor_idx.end()) {
					if (isOriginPure) {
						amper_list.push_back(_sensor_idx[a_sensor->_uuid]->_m->_idx);
						_log->debug() << "found the amper sensor(" + a_sensor->_uuid + ") in Pure position in new test";
					}
					else {
						amper_list.push_back(_sensor_idx[a_sensor->_uuid]->_cm->_idx);
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
				if (snapshot->_sensor_idx.find(s->_uuid) != snapshot->_sensor_idx.end()) {
					//if the other sensor id also in the other snapshot
					//get the current sensor pair
					SensorPair *sensor_pair = _sensor_pairs[ind(i, j)];
					//get the other 'same' sensor
					Sensor *c_s = snapshot->_sensor_idx[s->_uuid];
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
Input: the signal of all measurable
*/
void Snapshot::pruning(vector<bool> &signal){
	_dm->copy_arrays_to_sensors(0, _sensors.size(), _sensors);
	_dm->copy_arrays_to_sensor_pairs(0, _sensors.size(), _sensor_pairs);
	//get converted sensor list, from measurable signal
	vector<bool> sensor_list = convert_signal_to_sensor(signal);
	int row_escape = 0;
	int total_escape = 0;
	//destruct the corresponding sensors and sensor pairs
	for(int i = 0; i < _sensors.size(); ++i){
		if(sensor_list[i]){
			//delete the sensor if necessary
			_sensor_idx.erase(_sensors[i]->_m->_uuid);
			_sensor_idx.erase(_sensors[i]->_cm->_uuid);
			delete _sensors[i];
			_sensors[i] = NULL;
			row_escape++;
		}
		else{
			//or just adjust the idx of the sensor, and change the position
			_sensors[i]->setIdx(i - row_escape);
			_sensors[i - row_escape] = _sensors[i];
		}
		//delete the row of sensor, or the col in a row, where ther other sensor is deleted
		for(int j = 0; j <= i; ++j){
			if(sensor_list[i] || sensor_list[j]){
				//delete the sensor pair if necessary
				delete _sensor_pairs[ind(i, j)];
				_sensor_pairs[ind(i, j)] = NULL;
				total_escape++;
			}
			else{
				//or just change the position
				_sensor_pairs[ind(i, j) - total_escape] = _sensor_pairs[ind(i, j)];
			}
		}
	}

	//earse the additional space
	_sensors.erase(_sensors.end() - row_escape, _sensors.end());
	_sensor_pairs.erase(_sensor_pairs.end() - total_escape, _sensor_pairs.end());
	//adjust the size variables
	_dm->set_size(_sensors.size(), false);

	_dm->create_sensors_to_arrays_index(0, _sensors.size(), _sensors);
	_dm->create_sensor_pairs_to_arrays_index(0, _sensors.size(), _sensor_pairs);
	_dm->copy_sensors_to_arrays(0, _sensors.size(), _sensors);
	_dm->copy_sensor_pairs_to_arrays(0, _sensors.size(), _sensor_pairs);

	_log->info() << "Pruning done successful";
}

vector<bool> Snapshot::convert_signal_to_sensor(vector<bool> &signal){
	vector<bool> result;
	for(int i = 0; i < signal.size() / 2; ++i){
		if(signal[2 * i] || signal[2 * i + 1]) result.push_back(true);
		else result.push_back(false);
	}
	return result;
}

/*
This function is converting the list, from bool to int
*/
vector<int> Snapshot::convert_list(vector<bool> &list){
	vector<int> converted_list;
	for(int i = 0; i < list.size(); ++i){
		if(list[i]) converted_list.push_back(i);
	}
	return converted_list;
}

void Snapshot::ampers(vector<vector<bool> > &lists, vector<std::pair<string, string> > &id_pairs){
	_dm->copy_arrays_to_sensors(0, _sensors.size(), _sensors);
	_dm->copy_arrays_to_sensor_pairs(0, _sensors.size(), _sensor_pairs);
	int success_amper = 0;
	//record how many delay are successful
	
	for(int i = 0; i < lists.size(); ++i){
		vector<int> list = convert_list(lists[i]);
		if (list.size() < 2) {
			_log->warn() << "The amper vector size is less than 2, will abort this amper operation, list id: " + to_string(i);
			continue;
		}
		amper(list, id_pairs[i]);
		success_amper++;
	}

	_log->info() << to_string(success_amper) + " out of " + to_string(lists.size()) + " amper successfully done";

	if(_sensors.size() > _dm->_sensor_size_max){
		_log->info() << "New sensor size larger than current max, will resize";
		//if need to reallocate
		_dm->reallocate_memory(_sensors.size());
		//copy every sensor back, since the memory is new
		_dm->create_sensors_to_arrays_index(0, _sensors.size(), _sensors);
		_dm->create_sensor_pairs_to_arrays_index(0, _sensors.size(), _sensor_pairs);
		_dm->copy_sensors_to_arrays(0, _sensors.size(), _sensors);
		_dm->copy_sensor_pairs_to_arrays(0, _sensors.size(), _sensor_pairs);
	}
	else{
		//else just update the actual size
		_dm->set_size(_sensors.size(), false);
		//copy just the new added sensors and sensor pairs
		_dm->create_sensors_to_arrays_index(_sensors.size() - success_amper, _sensors.size(), _sensors);
		_dm->create_sensor_pairs_to_arrays_index(_sensors.size() - success_amper, _sensors.size(), _sensor_pairs);
		_dm->copy_sensors_to_arrays(_sensors.size() - success_amper, _sensors.size(), _sensors);
		_dm->copy_sensor_pairs_to_arrays(_sensors.size() - success_amper, _sensors.size(), _sensor_pairs);
	}
}

//the input list size should be larger than 2
void Snapshot::amper(vector<int> &list, std::pair<string, string> &uuid) {
	if (list.size() < 2) {
		_log->warn() << "Amper list size is smaller than 2, will not continue";
	}
	try{
		amperand(list[1], list[0], true, uuid);
		for(int j = 2; j < list.size(); ++j){
			amperand(_sensors.back()->_m->_idx, list[j], false, uuid);
		}
	}
	catch(CoreException &e){
		throw CoreException("Fatal error while doing amper and", CoreException::FATAL, status_codes::ServiceUnavailable);
	}
}

void Snapshot::delays(vector<vector<bool> > &lists, vector<std::pair<string, string> > &id_pairs) {
	_dm->copy_arrays_to_sensors(0, _sensors.size(), _sensors);
	_dm->copy_arrays_to_sensor_pairs(0, _sensors.size(), _sensor_pairs);
	int success_delay = 0;
	//record how many delay are successful
	for(int i = 0; i < lists.size(); ++i){
		vector<int> list = convert_list(lists[i]);
		if (list.size() < 1) {
			_log->warn() << "The amper vector size is less than 1, will abort this amper operation, list id: " + to_string(i);
			continue;
		}
		if(list.size() == 1){
			try {
				generate_delayed_weights(list[0], true, id_pairs[i]);
			}
			catch (CoreException &e) {
				throw CoreException("Fatal error in generate_delayed_weights", CoreException::FATAL, status_codes::ServiceUnavailable);
			}
		}
		else{
			amper(list, id_pairs[i]);
			try {
				generate_delayed_weights(_sensors.back()->_m->_idx, false, id_pairs[i]);
			}
			catch (CoreException &e) {
				throw CoreException("Fatal error in generate_delayed_weights", CoreException::FATAL, status_codes::ServiceUnavailable);
			}
		}
		success_delay++;
		_log->info() << "A delayed sensor is generated " + id_pairs[i].first;
		string delay_list="";
		for(int j = 0; j < list.size(); ++j) delay_list += (to_string(list[j]) + ",");
		_log->verbose() << "The delayed sensor generated from " + delay_list;
	}

	_log->info() << to_string(success_delay) + " out of " + to_string(lists.size()) + " delay successfully done";

	if (_sensors.size() > _dm->_sensor_size_max) {
		_log->info() << "New sensor size larger than current max, will resize";
		//if need to reallocate
		_dm->reallocate_memory(_sensors.size());
		//copy every sensor back, since the memory is new
		_dm->create_sensors_to_arrays_index(0, _sensors.size(), _sensors);
		_dm->create_sensor_pairs_to_arrays_index(0, _sensors.size(), _sensor_pairs);
		_dm->copy_sensors_to_arrays(0, _sensors.size(), _sensors);
		_dm->copy_sensor_pairs_to_arrays(0, _sensors.size(), _sensor_pairs);
	}
	else {
		//else just update the actual size
		_dm->set_size(_sensors.size(), false);
		//copy just the new added sensors and sensor pairs
		_dm->create_sensors_to_arrays_index(_sensors.size() - success_delay, _sensors.size(), _sensors);
		_dm->create_sensor_pairs_to_arrays_index(_sensors.size() - success_delay, _sensors.size(), _sensor_pairs);
		_dm->copy_sensors_to_arrays(_sensors.size() - success_delay, _sensors.size(), _sensors);
		_dm->copy_sensor_pairs_to_arrays(_sensors.size() - success_delay, _sensors.size(), _sensor_pairs);
	}
}

//*********************************The folloing get/set function may change under REST Call infra***********************

/*
------------------------------------SET FUNCTION------------------------------------
*/
void Snapshot::setThreshold(double &threshold) {
	_threshold = threshold;
	_log->info() << "snapshot threshold changed to " + to_string(threshold);
}

void Snapshot::setQ(double &q) {
	_q = q;
	_log->info() << "snapshot q changed to " + to_string(q);
}

void Snapshot::setAutoTarget(bool &auto_target) {
	_auto_target = auto_target;
}

void Snapshot::setPropagateMask(bool &propagate_mask) {
	_propagate_mask = propagate_mask;
}

void Snapshot::setInitialSize(int &initial_size) {
	_initial_size = initial_size;
}

/*
------------------------------------SET FUNCTION------------------------------------
*/

/*
------------------------------------GET FUNCTION------------------------------------
*/

/*
this function is getting the measurable, from the sensor list
*/
Measurable *Snapshot::getMeasurable(int idx){
	int s_idx = idx / 2;
	if (s_idx >= _sensors.size() || s_idx <0) {
		throw CoreException("the input measurable index is out of range, input is " + to_string(s_idx) + " sensor num is" + to_string(idx), CoreException::ERROR, status_codes::BadRequest);
	}
	if(idx % 2 == 0){
		return _sensors[s_idx]->_m;
	}
	else{
		return _sensors[s_idx]->_cm;
	}
}

Measurable *Snapshot::getMeasurable(string &measurable_id) {
	Sensor *sensor = getSensor(measurable_id);
	if (measurable_id == sensor->_m->_uuid) return sensor->_m;
	else if(measurable_id == sensor->_cm->_uuid) return sensor->_cm;
	throw CoreException("Cannot find the measurable id " + measurable_id, CoreException::FATAL, status_codes::ServiceUnavailable);
}

SensorPair *Snapshot::getSensorPair(Sensor *sensor1, Sensor *sensor2) {
	int idx1 = sensor1->_idx > sensor2->_idx ? sensor1->_idx : sensor2->_idx;
	int idx2 = sensor1->_idx > sensor2->_idx ? sensor2->_idx : sensor1->_idx;
	return _sensor_pairs[ind(idx1, idx2)];
}

/*
This function is getting the measurable pair from the sensor pair list
Input: m_idx1, m_idx2 are index of the measurable, m_idx1 > m_idx2
*/
MeasurablePair *Snapshot::getMeasurablePair(int m_idx1, int m_idx2){
	int s_idx1 = m_idx1 / 2;
	int s_idx2 = m_idx2 / 2;
	Measurable *m1 = getMeasurable(m_idx1);
	Measurable *m2 = getMeasurable(m_idx2);
	return _sensor_pairs[ind(s_idx1, s_idx2)]->getMeasurablePair(m1->_isOriginPure, m2->_isOriginPure);
}

MeasurablePair *Snapshot::getMeasurablePair(string &mid1, string &mid2) {
	int idx1 = getMeasurable(mid1)->_idx > getMeasurable(mid2)->_idx ? getMeasurable(mid1)->_idx : getMeasurable(mid2)->_idx;
	int idx2 = getMeasurable(mid1)->_idx > getMeasurable(mid2)->_idx ? getMeasurable(mid2)->_idx : getMeasurable(mid1)->_idx;
	return getMeasurablePair(idx1, idx2);
}

vector<bool> Snapshot::getAmperList(string &sensor_id) {
	if (_sensor_idx.find(sensor_id) == _sensor_idx.end()) {
		throw CoreException("Cannot find the sensor id " + sensor_id, CoreException::ERROR, status_codes::NotFound);
	}
	Sensor *sensor = _sensor_idx[sensor_id];
	vector<bool> result(_dm->_measurable_size, false);
	for (int i = 0; i < sensor->_amper.size(); ++i) {
		result[sensor->_amper[i]] = true;
	}
	return result;
}

vector<string> Snapshot::getAmperListID(string &sensor_id) {
	if (_sensor_idx.find(sensor_id) == _sensor_idx.end()) {
		throw CoreException("Cannot find the sensor id " + sensor_id, CoreException::ERROR, status_codes::NotFound);
	}
	Sensor *sensor = _sensor_idx[sensor_id];
	vector<string> result;
	for (int i = 0; i < sensor->_amper.size(); ++i) {
		int idx = sensor->_amper[i];
		Sensor *s = _sensors[idx / 2];
		if (idx % 2 == 0) result.push_back(s->_m->_uuid);
		else result.push_back(s->_cm->_uuid);
	}
	return result;
}

Sensor *Snapshot::getSensor(string &sensor_id) {
	if (_sensor_idx.find(sensor_id) == _sensor_idx.end()) {
		throw CoreException("Cannot find the sensor id " + sensor_id, CoreException::ERROR, status_codes::NotFound);
	}
	return _sensor_idx[sensor_id];
}

vector<std::pair<int, pair<string, string> > > Snapshot::getSensorInfo() {
	vector<std::pair<int, pair<string, string> > > results;
	for (int i = 0; i < _sensors.size(); ++i) {
		Sensor *sensor = _sensors[i];
		std::pair<string, string> sensor_pair(_sensors[i]->_m->_uuid, _sensors[i]->_cm->_uuid);
		results.push_back(std::pair<int, pair<string, string> >(sensor->_idx, sensor_pair));
	}
	return results;
}

double Snapshot::getTotal() {
	return _total;
}

double Snapshot::getQ() {
	return _q;
}

double Snapshot::getThreshold() {
	return _threshold;
}

bool Snapshot::getAutoTarget() {
	return _auto_target;
}

bool Snapshot::getPropagateMask() {
	return _propagate_mask;
}

int Snapshot::getInitialSize() {
	return _initial_size;
}

/*
------------------------------------GET FUNCTION------------------------------------
*/


void Snapshot::create_implication(string &sensor1, string &sensor2) {
	if (_sensor_idx.find(sensor1) == _sensor_idx.end()) {
		_log->error() << "Cannot find the sensor " + sensor1;
		throw CoreException("Cannot find the sensor " + sensor1, CoreException::ERROR, status_codes::BadRequest);
	}
	if (_sensor_idx.find(sensor2) == _sensor_idx.end()) {
		_log->error() << "Cannot find the sensor " + sensor2;
		throw CoreException("Cannot find the sensor " + sensor2, CoreException::ERROR, status_codes::BadRequest);
	}
	int idx1 = sensor1 == _sensor_idx[sensor1]->_m->_uuid ? _sensor_idx[sensor1]->_m->_idx : _sensor_idx[sensor1]->_cm->_idx;
	int idx2 = sensor2 == _sensor_idx[sensor2]->_m->_uuid ? _sensor_idx[sensor2]->_m->_idx : _sensor_idx[sensor2]->_cm->_idx;
	_dm->set_implication(true, idx1, idx2);
	_log->info() << "Implication success from " + sensor1 + " to " + sensor2;
}

void Snapshot::delete_implication(string &sensor1, string &sensor2) {
	if (_sensor_idx.find(sensor1) == _sensor_idx.end()) {
		_log->error() << "Cannot find the sensor " + sensor1;
		throw CoreException("Cannot find the sensor " + sensor1, CoreException::ERROR, status_codes::BadRequest);
	}
	if (_sensor_idx.find(sensor2) == _sensor_idx.end()) {
		_log->error() << "Cannot find the sensor " + sensor2;
		throw CoreException("Cannot find the sensor " + sensor2, CoreException::ERROR, status_codes::BadRequest);
	}
	int idx1 = sensor1 == _sensor_idx[sensor1]->_m->_uuid ? _sensor_idx[sensor1]->_m->_idx : _sensor_idx[sensor1]->_cm->_idx;
	int idx2 = sensor2 == _sensor_idx[sensor2]->_m->_uuid ? _sensor_idx[sensor2]->_m->_idx : _sensor_idx[sensor2]->_cm->_idx;
	_dm->set_implication(false, idx1, idx2);
	_log->info() << "Implication delete success from " + sensor1 + " to " + sensor2;
}

bool Snapshot::get_implication(string &sensor1, string &sensor2) {
	if (_sensor_idx.find(sensor1) == _sensor_idx.end()) {
		_log->error() << "Cannot find the sensor " + sensor1;
		throw CoreException("Cannot find the sensor " + sensor1, CoreException::ERROR, status_codes::BadRequest);
	}
	if (_sensor_idx.find(sensor2) == _sensor_idx.end()) {
		_log->error() << "Cannot find the sensor " + sensor2;
		throw CoreException("Cannot find the sensor " + sensor2, CoreException::ERROR, status_codes::BadRequest);
	}
	int idx1 = sensor1 == _sensor_idx[sensor1]->_m->_uuid ? _sensor_idx[sensor1]->_m->_idx : _sensor_idx[sensor1]->_cm->_idx;
	int idx2 = sensor2 == _sensor_idx[sensor2]->_m->_uuid ? _sensor_idx[sensor2]->_m->_idx : _sensor_idx[sensor2]->_cm->_idx;
	bool value = _dm->get_implication(idx1, idx2);
	return value;
}


void Snapshot::delete_sensor(string &sensor_id) {
	if (_sensor_idx.find(sensor_id) == _sensor_idx.end()) {
		_log->error() << "Cannot find the sensor " + sensor_id;
		throw CoreException("Cannot find the sensor " + sensor_id, CoreException::ERROR, status_codes::BadRequest);
	}

	int sensor_idx = _sensor_idx[sensor_id]->_idx;
	vector<bool> pruning_list(_dm->_measurable_size, false);
	pruning_list[2 * sensor_idx] = true;
	pruning(pruning_list);
	_log->info() << "Sensor " + sensor_id + " deleted";
}

/*
This is the amper and function, used in amper and delay
Input: m_idx1, m_idx2 are the measurable idx that need to be amperand, m_idx1 > m_idx2, merge is indicating whether merge or replace the last sensor/row of sensor pair
*/
void Snapshot::amperand(int m_idx1, int m_idx2, bool merge, std::pair<string, string> &id_pair) {
	vector<SensorPair*> amper_and_sensor_pairs;
	Sensor *amper_and_sensor = new Sensor(id_pair, _total, _sensors.size());
	_sensor_idx[id_pair.first] = amper_and_sensor;
	_sensor_idx[id_pair.second] = amper_and_sensor;

	double f = 1.0;
	if(_total > 1e-5)
		f = getMeasurablePair(m_idx1, m_idx2)->v_w / _total;
	for(int i = 0; i < _sensors.size(); ++i){
		SensorPair *sensor_pair = NULL;
		sensor_pair = new SensorPair(amper_and_sensor, _sensors[i], _threshold, _total);
		Measurable *m1 = _sensors[i]->_m;
		Measurable *m2 = _sensors[i]->_cm;

		if (_sensors[i]->_m->_idx == m_idx1 || _sensors[i]->_m->_idx == m_idx2) {
			sensor_pair->mij->v_w = getMeasurablePair(m_idx1, m_idx2)->v_w;
			sensor_pair->mi_j->v_w = 0.0;
			if (_sensors[i]->_m->_idx == m_idx1) {
				sensor_pair->m_ij->v_w = getMeasurablePair(m_idx1, compi(m_idx2))->v_w;
				sensor_pair->m_i_j->v_w = getMeasurablePair(compi(m_idx1), compi(m_idx1))->v_w;
			}
			else {
				sensor_pair->m_ij->v_w = getMeasurablePair(compi(m_idx1), m_idx2)->v_w;
				sensor_pair->m_i_j->v_w = getMeasurablePair(compi(m_idx2), compi(m_idx2))->v_w;
			}
		}
		else if (_sensors[i]->_cm->_idx == m_idx1 || _sensors[i]->_cm->_idx == m_idx2) {
			sensor_pair->mi_j->v_w = getMeasurablePair(m_idx1, m_idx2)->v_w;
			sensor_pair->mij->v_w = 0.0;
			if (_sensors[i]->_cm->_idx == m_idx1) {
				sensor_pair->m_i_j->v_w = getMeasurablePair(m_idx1, compi(m_idx2))->v_w;
				sensor_pair->m_ij->v_w = getMeasurablePair(compi(m_idx1), compi(m_idx1))->v_w;
			}
			else {
				sensor_pair->m_i_j->v_w = getMeasurablePair(compi(m_idx1), m_idx2)->v_w;
				sensor_pair->m_ij->v_w = getMeasurablePair(compi(m_idx2), compi(m_idx2))->v_w;
			}
		}
		else {
			sensor_pair->mij->v_w = f * getMeasurablePair(m1->_idx, m1->_idx)->v_w;
			sensor_pair->mi_j->v_w = f * getMeasurablePair(m2->_idx, m2->_idx)->v_w;
			sensor_pair->m_ij->v_w = (1.0 - f) * getMeasurablePair(m1->_idx, m1->_idx)->v_w;
			sensor_pair->m_i_j->v_w = (1.0 - f) * getMeasurablePair(m2->_idx, m2->_idx)->v_w;
		}
		amper_and_sensor_pairs.push_back(sensor_pair);
	}
	SensorPair *self_pair = new SensorPair(amper_and_sensor, amper_and_sensor, _threshold, _total);
	self_pair->mij->v_w = amper_and_sensor_pairs[0]->mij->v_w + amper_and_sensor_pairs[0]->mi_j->v_w;
	self_pair->mi_j->v_w = 0.0;
	self_pair->m_ij->v_w = 0.0;
	self_pair->m_i_j->v_w = amper_and_sensor_pairs[0]->m_ij->v_w + amper_and_sensor_pairs[0]->m_i_j->v_w;
	amper_and_sensor_pairs.push_back(self_pair);
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
			delete _sensor_pairs[i];
		}
		_sensor_pairs.erase(_sensor_pairs.begin() + ind(_sensors.size(), 0), _sensor_pairs.end());
		//also need to remove the n-1 position of amper_and_sensor_pairs
		delete amper_and_sensor_pairs[_sensors.size()];
		amper_and_sensor_pairs.erase(amper_and_sensor_pairs.end() - 2, amper_and_sensor_pairs.end() - 1);
	}
	else{
		//set the amper list, if append a new sensor, append the smaller idx first
		amper_and_sensor->setAmperList(m_idx2);
		amper_and_sensor->setAmperList(m_idx1);
	}
	_sensors.push_back(amper_and_sensor);
	_sensor_pairs.insert(_sensor_pairs.end(), amper_and_sensor_pairs.begin(), amper_and_sensor_pairs.end());
}

/*
This function is generating the delayed weights
Before this function is called, have to make sure, _sensors and _sensor_pairs have valid info in vwxx;
Input: mid of the measurable doing the delay, and whether to merge after the operation
*/
void Snapshot::generate_delayed_weights(int mid, bool merge, std::pair<string, string> &id_pair){
	//create a new delayed sensor
	Sensor *delayed_sensor = new Sensor(id_pair, _total, _sensors.size());
	_sensor_idx[id_pair.first] = delayed_sensor;
	_sensor_idx[id_pair.second] = delayed_sensor;
	vector<SensorPair *> delayed_sensor_pairs;
	//the sensor name is TBD, need python input
	int delay_mid1 = mid, delay_mid2 = compi(mid);
	int sid = mid / 2;
	//get mid and sid

	bool is_sensor_active;
	if(merge){
		//if need to merge, means just a single sensor delay
		is_sensor_active = _sensors[sid]->isSensorActive();
	}
	else{
		//means not a single sensor delay
		is_sensor_active = amper_and_signals(_sensors[sid]);
	}
	//reverse for compi
	if (mid % 2 == 1) is_sensor_active = !is_sensor_active;

	for(int i = 0; i < _sensors.size() + 1; ++i){
		SensorPair *sensor_pair = NULL;
		if(i == _sensors.size()){
			//if this is the last sensor pair, and it is the pair of the delayed sensor itself
			sensor_pair = new SensorPair(delayed_sensor, delayed_sensor, _threshold, _total);
			//copy all those diag values first
			delayed_sensor->_m->_vdiag = delayed_sensor_pairs[0]->mij->v_w + delayed_sensor_pairs[0]->mi_j->v_w;
			delayed_sensor->_cm->_vdiag = delayed_sensor_pairs[0]->m_ij->v_w + delayed_sensor_pairs[0]->m_i_j->v_w;
			delayed_sensor->_m->_vdiag_ = delayed_sensor->_m->_vdiag;
			delayed_sensor->_cm->_vdiag_ = delayed_sensor->_cm->_vdiag;
			//then assign the value to sensor pair
			sensor_pair->mij->v_w = delayed_sensor->_m->_vdiag;
			sensor_pair->mi_j->v_w = 0;
			sensor_pair->m_ij->v_w = 0;
			sensor_pair->m_i_j->v_w = delayed_sensor->_cm->_vdiag;
			delayed_sensor_pairs.push_back(sensor_pair);
		}
		else{
			sensor_pair = new SensorPair(delayed_sensor, _sensors[i], _threshold, _total);
			sensor_pair->mij->v_w = is_sensor_active * _sensors[i]->_m->_vdiag;
			sensor_pair->mi_j->v_w = is_sensor_active * _sensors[i]->_cm->_vdiag;
			sensor_pair->m_ij->v_w = !is_sensor_active * _sensors[i]->_m->_vdiag;
			sensor_pair->m_i_j->v_w = !is_sensor_active * _sensors[i]->_cm->_vdiag;
			delayed_sensor_pairs.push_back(sensor_pair);
		}
	}

	if(!merge){
		//replace the last one
		Sensor *old_sensor = _sensors[sid];
		//set the mask amper of the delayed sensor to be the same as the one that need to be delayed
		delayed_sensor->setAmperList(old_sensor);
		//take the idx of the old one, copy it to the new one 
		delayed_sensor->setIdx(old_sensor->_idx);
		//delete the old one
		delete old_sensor;
		//if no mrege needed, means the last existing sensor and row of sensor_pair need to be removed
		_sensors.pop_back();
		//destruct the last row of sensor pair
		for(int i = ind(_sensors.size(), 0); i < ind(_sensors.size() + 1, 0); ++i){
			delete _sensor_pairs[i];
		}
		_sensor_pairs.erase(_sensor_pairs.begin() + ind(_sensors.size(), 0), _sensor_pairs.end());
		//also need to remove the n-1 position of delayed_sensor_pair
		delete delayed_sensor_pairs[_sensors.size()];
		delayed_sensor_pairs.erase(delayed_sensor_pairs.end() - 2, delayed_sensor_pairs.end() - 1);
	}
	else{
		delayed_sensor->setAmperList(mid);
	}
	_sensors.push_back(delayed_sensor);
	_sensor_pairs.insert(_sensor_pairs.end(), delayed_sensor_pairs.begin(), delayed_sensor_pairs.end());
}

/*
This function is using the amper and observe array to get the delayed sensor value
Input: observe array
*/
bool Snapshot::amper_and_signals(Sensor *sensor) {
	vector<int> amper_list = sensor->getAmperList();
	for (int i = 0; i < amper_list.size(); ++i) {
		int j = amper_list[i];
		Measurable *m = getMeasurable(j);
		if (!(m->getOldObserve())) return false;
	}
	return true;
}

DataManager *Snapshot::getDM() {
	return _dm;
}

/*
void Snapshot::save_snapshot(ofstream &file) {
	//write uuid
	int uuid_size = _uuid.length();
	file.write(reinterpret_cast<const char *>(&uuid_size), sizeof(int));
	file.write(_uuid.c_str(), uuid_size * sizeof(char));
	int sensor_size = _sensors.size();
	int sensor_pair_size = _sensor_pairs.size();
	file.write(reinterpret_cast<const char *>(&sensor_size), sizeof(int));
	file.write(reinterpret_cast<const char *>(&sensor_pair_size), sizeof(int));
	copy_arrays_to_sensors(0, sensor_size);
	copy_arrays_to_sensor_pairs(0, sensor_size);
	for (int i = 0; i < sensor_size; ++i) {
		_sensors[i]->save_sensor(file);
	}
	for (int i = 0; i < sensor_pair_size; ++i) {
		_sensor_pairs[i]->save_sensor_pair(file);
	}
}
*/
/*
----------------Snapshot Base Class-------------------
*/

/*
----------------Snapshot_Stationary Class-------------------
*/
//Snapshot_Stationary::Snapshot_Stationary(ifstream &file, string &log_dir):Snapshot(file, log_dir) {}

Snapshot_Stationary::Snapshot_Stationary(string uuid, string log_dir)
	:Snapshot(uuid, log_dir){
}

void Snapshot_Stationary::update_total(double phi, bool active) {
	_total_ = _total;
	if (active) {
		_total = _q * _total + (1 - _q) * phi;
	}
}

Snapshot_Stationary::~Snapshot_Stationary(){}

/*
----------------Snapshot_Stationary Class-------------------
*/


/*
----------------Snapshot_Forgetful Class-------------------
*/
/*
Snapshot_Forgetful::Snapshot_Forgetful(string uuid, string log_dir)
	:Snapshot(uuid, log_dir){
}

void Snapshot_Forgetful::update_total(double phi, bool active) {
	Snapshot::calculate_total(phi, active);
}

Snapshot_Forgetful::~Snapshot_Forgetful(){}
*/
/*
----------------Snapshot_Forgetful Class-------------------
*/
