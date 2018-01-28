#include "DataManager.h"
#include "World.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "UMAException.h"
#include "Logger.h"
#include "data_util.h"
#include "uma_base.cuh"
#include "kernel_util.cuh"

extern int ind(int row, int col);
extern int compi(int x);
static Logger dataManagerLogger("DataManager", "log/dataManager.log");
/*
init function
Input: DataManager's log path
*/
DataManager::DataManager(const string &dependency): _dependency(dependency) {
	_memory_expansion = stod(World::core_info["DataManager"]["memory_expansion"]);

	init_pointers();
	set_size(0);
	dataManagerLogger.info("Setting the memory expansion rate to " + to_string(_memory_expansion), _dependency);
}

DataManager::~DataManager() {
}

/*
init all pointers, to be NULL
*/
void DataManager::init_pointers() {
	h_dirs = NULL;
	h_weights = NULL;
	h_thresholds = NULL;
	h_mask_amper = NULL;
	h_npdirs = NULL;
	h_observe = NULL;
	h_observe_ = NULL;
	h_signal = NULL;
	h_load = NULL;
	h_current = NULL;
	h_mask = NULL;
	h_target = NULL;
	h_prediction = NULL;
	h_diag = NULL;
	h_diag_ = NULL;
	h_up = NULL;
	h_down = NULL;
	h_npdir_mask = NULL;
	h_signals = NULL;
	h_dists = NULL;
	h_negligible = NULL;
	h_union_root = NULL;
	h_res = NULL;

	dev_dirs = NULL;
	dev_weights = NULL;
	dev_thresholds = NULL;
	dev_mask_amper = NULL;
	dev_npdirs = NULL;
	dev_observe = NULL;
	dev_signal = NULL;
	dev_load = NULL;
	dev_current = NULL;
	dev_mask = NULL;
	dev_target = NULL;
	dev_diag = NULL;
	dev_diag_ = NULL;
	dev_prediction = NULL;
	dev_npdir_mask = NULL;
	dev_signals = NULL;
	dev_dists = NULL;
	dev_negligible = NULL;
	dev_union_root = NULL;
	dev_res = NULL;

	dataManagerLogger.debug("Setting all pointers to NULL", _dependency);
}

/*
Remalloc the memory, based on the sensor size
Input: the sensor size
*/
void DataManager::reallocate_memory(double &total, int sensor_size) {
	dataManagerLogger.info("Starting reallocating memory", _dependency);
	//free all memory first
	free_all_parameters();
	//then set the size to be new one
	set_size(sensor_size, true);

	try {
		//then generate all matrix again
		gen_weight();
		gen_direction();
		gen_thresholds();
		gen_mask_amper();
		gen_np_direction();
		gen_signals();
		gen_npdir_mask();
		gen_dists();
		gen_other_parameters();
	}
	catch (UMAException &e) {
		dataManagerLogger.error("Fatal error in reallocate_memory when doing memory allocation", _dependency);
		throw UMAException("Fatal error in reallocate_memory when doing memory allocation", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
	}

	try {
		//init other parameters
		init_other_parameter(total);
	}
	catch (UMAException &e) {
		dataManagerLogger.error("Fatal error in reallocate_memory when doing parameters init", _dependency);
		throw UMAException("Fatal error in reallocate_memory when doing parameters ini", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
	}
	dataManagerLogger.info("Memory reallocated!", _dependency);
}

/*
Set the size
Input: new sensor size and whether changing max of not
*/
void DataManager::set_size(int sensor_size, bool change_max) {
	_sensor_size = sensor_size;
	_measurable_size = 2 * _sensor_size;
	_sensor2d_size = _sensor_size * (_sensor_size + 1) / 2;
	_measurable2d_size = _measurable_size * (_measurable_size + 1) / 2;
	_mask_amper_size = _sensor_size * (_sensor_size + 1);
	_npdir_size = _measurable2d_size + _sensor_size;

	dataManagerLogger.info("Setting sensor size to " + to_string(_sensor_size), _dependency);
	dataManagerLogger.debug("Setting measurable size to " + to_string(_measurable_size), _dependency);
	dataManagerLogger.debug("Setting sensor size 2D to " + to_string(_sensor2d_size), _dependency);
	dataManagerLogger.debug("Setting measurable size 2D to " + to_string(_measurable2d_size), _dependency);
	dataManagerLogger.debug("Setting mask amper size to " + to_string(_mask_amper_size), _dependency);
	dataManagerLogger.debug("Setting npdir size to " + to_string(_npdir_size), _dependency);

	if (change_max) {
		_sensor_size_max = (int)(_sensor_size * (1 + _memory_expansion));
		_measurable_size_max = 2 * _sensor_size_max;
		_sensor2d_size_max = _sensor_size_max * (_sensor_size_max + 1) / 2;
		_measurable2d_size_max = _measurable_size_max * (_measurable_size_max + 1) / 2;
		_mask_amper_size_max = _sensor_size_max * (_sensor_size_max + 1);
		_npdir_size_max = _measurable2d_size_max + _sensor_size_max;

		dataManagerLogger.debug("Setting max sensor size to " + to_string(_sensor_size_max), _dependency);
		dataManagerLogger.debug("Setting max measurable size to " + to_string(_measurable_size_max), _dependency);
		dataManagerLogger.debug("Setting max sensor size 2D to " + to_string(_sensor2d_size_max), _dependency);
		dataManagerLogger.debug("Setting max measurable size 2D to " + to_string(_measurable2d_size_max), _dependency);
		dataManagerLogger.debug("Setting max mask amper size to " + to_string(_mask_amper_size_max), _dependency);
		dataManagerLogger.debug("Setting max npdir size to " + to_string(_npdir_size_max), _dependency);
	}
	else dataManagerLogger.info("All size max value remain the same", _dependency);
}

/*
init other parameters
Input: total value
*/
void DataManager::init_other_parameter(double &total) {
	for (int i = 0; i < _measurable_size_max; ++i) {
		h_observe[i] = false;
		h_observe_[i] = false;
		h_signal[i] = false;
		h_load[i] = false;
		h_mask[i] = false;
		h_current[i] = false;
		h_target[i] = false;
		h_prediction[i] = false;
		h_up[i] = false;
		h_down[i] = false;
		h_diag[i] = total / 2.0;
		h_diag_[i] = total / 2.0;
		h_negligible[i] = false;
	}
	*h_res = 0.0;
	for (int i = 0; i < _sensor_size_max; ++i) {
		h_union_root[i] = 0;
	}

	data_util::dev_init(dev_observe, _measurable_size_max);
	data_util::dev_init(dev_signal, _measurable_size_max);
	data_util::dev_init(dev_load, _measurable_size_max);
	data_util::dev_init(dev_mask, _measurable_size_max);
	data_util::dev_init(dev_current, _measurable_size_max);
	data_util::dev_init(dev_target, _measurable_size_max);
	data_util::dev_init(dev_prediction, _measurable_size_max);
	data_util::dev_init(dev_negligible, _measurable_size_max);
	data_util::dev_init(dev_res, 1);
	data_util::dev_init(dev_union_root, _sensor_size_max);

	uma_base::init_diag(dev_diag, dev_diag_, total, total, _measurable_size_max);
}


/*
This function free data together
Deprecated in new version
Input: None
Output: None
*/
void DataManager::free_all_parameters() {//free data in case of memory leak
	dataManagerLogger.info("Starting releasing memory", _dependency);
	try {
		delete[] h_dirs;
		delete[] h_weights;
		delete[] h_thresholds;
		delete[] h_mask_amper;
		delete[] h_observe;
		delete[] h_observe_;
		delete[] h_signal;
		delete[] h_load;
		delete[] h_mask;
		delete[] h_current;
		delete[] h_target;
		delete[] h_negligible;
		delete[] h_diag;
		delete[] h_diag_;
		delete[] h_prediction;
		delete[] h_up;
		delete[] h_down;
		delete[] h_npdirs;
		delete[] h_npdir_mask;
		delete[] h_signals;
		delete[] h_dists;
		delete[] h_union_root;
		delete h_res;
	}
	catch (UMAException &e) {
		dataManagerLogger.error("Fatal error in free_all_parameters, when doing cpu array release", _dependency);
		throw UMAException("Fatal error in free_all_parameters, when doing cpu array release", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
	}

	try {
		data_util::dev_free(dev_dirs);
		data_util::dev_free(dev_weights);
		data_util::dev_free(dev_thresholds);
		data_util::dev_free(dev_mask_amper);

		data_util::dev_free(dev_mask);
		data_util::dev_free(dev_current);
		data_util::dev_free(dev_target);

		data_util::dev_free(dev_observe);
		data_util::dev_free(dev_signal);
		data_util::dev_free(dev_load);

		data_util::dev_free(dev_prediction);
		data_util::dev_free(dev_negligible);
		data_util::dev_free(dev_diag);
		data_util::dev_free(dev_diag_);

		data_util::dev_free(dev_npdir_mask);
		data_util::dev_free(dev_npdirs);
		data_util::dev_free(dev_signals);
		data_util::dev_free(dev_dists);
		data_util::dev_free(dev_union_root);

		data_util::dev_free(dev_res);
	}
	catch (UMAException &e) {
		dataManagerLogger.error("Fatal error in free_all_parameters, when doing gpu array release", _dependency);
		throw UMAException("Fatal error in free_all_parameters, when doing gpu array release", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
	}
	dataManagerLogger.info("All memory released", _dependency);
}


/*
This function is generating dir matrix memory both on host and device
*/
void DataManager::gen_direction() {
	//malloc the space
	h_dirs = new bool[_measurable2d_size_max];
	data_util::dev_bool(dev_dirs, _measurable2d_size_max);

	//fill in all 0
	memset(h_dirs, 0, _measurable2d_size_max * sizeof(bool));
	data_util::dev_init(dev_dirs, _measurable2d_size_max);

	dataManagerLogger.debug("Dir matrix generated with size " + to_string(_measurable2d_size_max), _dependency);
}

/*
This function is generating weight matrix memory both on host and device
*/
void DataManager::gen_weight() {
	//malloc the space
	h_weights = new double[_measurable2d_size_max];
	data_util::dev_double(dev_weights, _measurable2d_size_max);

	//fill in all 0
	memset(h_weights, 0, _measurable2d_size_max * sizeof(double));
	data_util::dev_init(dev_weights, _measurable2d_size_max);

	dataManagerLogger.debug("Weight matrix generated with size " + to_string(_measurable2d_size_max), _dependency);
}

/*
This function is generating threshold matrix memory both on host and device
*/
void DataManager::gen_thresholds() {
	//malloc the space
	h_thresholds = new double[_sensor2d_size_max];
	data_util::dev_double(dev_thresholds, _sensor2d_size_max);

	//fill in all 0
	memset(h_thresholds, 0, _sensor2d_size_max * sizeof(double));
	data_util::dev_init(dev_thresholds, _sensor2d_size_max);

	dataManagerLogger.debug("Threshold matrix generated with the size " + to_string(_sensor2d_size_max), _dependency);
}

/*
This function is generating mask amper matrix memory both on host and device
*/
void DataManager::gen_mask_amper() {
	//malloc the space
	h_mask_amper = new bool[_mask_amper_size_max];
	data_util::dev_bool(dev_mask_amper, _mask_amper_size_max);

	//fill in all 0
	memset(h_mask_amper, 0, _mask_amper_size_max * sizeof(bool));
	data_util::dev_init(dev_mask_amper, _mask_amper_size_max);

	dataManagerLogger.debug("Mask amper matrix generated with size " + to_string(_mask_amper_size_max), _dependency);
}

/*
This function is generating n power of dir matrix both on host and device
*/
void DataManager::gen_np_direction() {
	//malloc the space
	h_npdirs = new bool[_npdir_size_max];
	data_util::dev_bool(dev_npdirs, _npdir_size_max);

	//fill in all 0
	memset(h_npdirs, 0, _npdir_size_max * sizeof(bool));
	data_util::dev_init(dev_npdirs, _npdir_size_max);

	dataManagerLogger.debug("NPDir matrix generated with size " + to_string(_npdir_size_max), _dependency);
}

void DataManager::gen_signals() {
	//malloc the space
	h_signals = new bool[_measurable_size_max * _measurable_size_max];
	h_lsignals = new bool[_measurable_size_max * _measurable_size_max];
	data_util::dev_bool(dev_signals, _measurable_size_max * _measurable_size_max);
	data_util::dev_bool(dev_lsignals, _measurable_size_max * _measurable_size_max);

	//init with all false
	memset(h_signals, 0, _measurable_size_max * _measurable_size_max * sizeof(bool));
	memset(h_lsignals, 0, _measurable_size_max * _measurable_size_max * sizeof(bool));
	data_util::dev_init(dev_signals, _measurable_size_max * _measurable_size_max);
	data_util::dev_init(dev_lsignals, _measurable_size_max * _measurable_size_max);

	dataManagerLogger.debug(to_string(_measurable_size_max) + " num of signals with length " + to_string(_measurable_size_max) + " are generated, total size " + to_string(_measurable_size_max * _measurable_size_max), _dependency);
	dataManagerLogger.debug(to_string(_measurable_size_max) + " num of loaded signals with length " + to_string(_measurable_size_max) + " are generated, total size " + to_string(_measurable_size_max * _measurable_size_max), _dependency);
}

void DataManager::gen_npdir_mask() {
	//malloc the space
	h_npdir_mask = new bool[_sensor_size_max * _measurable_size_max];
	data_util::dev_bool(dev_npdir_mask, _sensor_size_max * _measurable_size_max);

	//init with all false
	memset(h_npdir_mask, 0, _sensor_size_max * _measurable_size_max * sizeof(bool));
	data_util::dev_init(dev_npdir_mask, _sensor_size_max * _measurable_size_max);

	dataManagerLogger.debug(to_string(_sensor_size_max) + " num of npdir mask with length " + to_string(_measurable_size_max) + " are generated, total size " + to_string(_sensor_size_max * _measurable_size_max), _dependency);
}

void DataManager::gen_dists() {
	//malloc the space
	h_dists = new int[_measurable_size_max * _measurable_size_max];
	data_util::dev_int(dev_dists, _measurable_size_max * _measurable_size_max);

	//init with all 0
	memset(h_dists, 0, _measurable_size_max * _measurable_size_max * sizeof(int));
	data_util::dev_init(dev_dists, _measurable_size_max * _measurable_size_max);

	dataManagerLogger.debug(to_string(_measurable_size_max) + "*" + to_string(_measurable_size_max) + "=" + to_string(_measurable_size_max * _measurable_size_max) + " num of space allocated for dists, used for block GPU", _dependency);
}

/*
This function generate other parameter
*/
void DataManager::gen_other_parameters() {
	h_observe = new bool[_measurable_size_max];
	h_observe_ = new bool[_measurable_size_max];
	h_signal = new bool[_measurable_size_max];
	h_load = new bool[_measurable_size_max];
	h_mask = new bool[_measurable_size_max];
	h_current = new bool[_measurable_size_max];
	h_target = new bool[_measurable_size_max];
	h_negligible = new bool[_measurable_size_max];
	h_diag = new double[_measurable_size_max];
	h_diag_ = new double[_measurable_size_max];
	h_prediction = new bool[_measurable_size_max];
	h_up = new bool[_measurable_size_max];
	h_down = new bool[_measurable_size_max];
	h_union_root = new int[_sensor_size_max];
	h_res = new float;

	data_util::dev_bool(dev_observe, _measurable_size_max);
	data_util::dev_bool(dev_signal, _measurable_size_max);
	data_util::dev_bool(dev_load, _measurable_size_max);
	data_util::dev_bool(dev_mask, _measurable_size_max);
	data_util::dev_bool(dev_current, _measurable_size_max);
	data_util::dev_bool(dev_target, _measurable_size_max);
	data_util::dev_bool(dev_prediction, _measurable_size_max);
	data_util::dev_bool(dev_negligible, _measurable_size_max);

	data_util::dev_double(dev_diag, _measurable_size_max);
	data_util::dev_double(dev_diag_, _measurable_size_max);
	data_util::dev_int(dev_union_root, _sensor_size_max);
	data_util::dev_float(dev_res, 1);

	dataManagerLogger.debug("Other parameter generated, with size " + to_string(_measurable_size_max), _dependency);
}

void DataManager::create_sensors_to_arrays_index(const int start_idx, const int end_idx, const vector<Sensor*> &sensors) {
	//create idx for diag and current
	for (int i = start_idx; i < end_idx; ++i) {
		try {
			sensors[i]->setMeasurableDiagPointers(h_diag, h_diag_);
			sensors[i]->setMeasurableObservePointers(h_observe, h_observe_);
			sensors[i]->setMeasurableCurrentPointers(h_current);
		}
		catch (exception &e) {
			dataManagerLogger.error("Fatal error while doing create_sensor_to_arrays_index", _dependency);
			throw UMAException("Fatal error happen in create_sensors_to_arrays_index", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
		}
	}
	dataManagerLogger.info("Sensor from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " have created idx to arrays", _dependency);
}

void DataManager::create_sensor_pairs_to_arrays_index(const int start_idx, const int end_idx, const vector<SensorPair*> &sensor_pairs) {
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		try {
			sensor_pairs[i]->setAllPointers(h_weights, h_dirs, h_thresholds);
		}
		catch (exception &e) {
			dataManagerLogger.error("Fatal error while doing create_sensor_pairs_to_arrays_index", _dependency);
			throw UMAException("Fatal error happen in create_sensor_pairs_to_arrays_index", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
		}
	}
	dataManagerLogger.info("Sensor pairs from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " have created idx to arrays", _dependency);
}

/*
--------------------Sensor Object And Array Data Transfer---------------------
*/
void DataManager::copy_arrays_to_sensors(const int start_idx, const int end_idx, const vector<Sensor*> &_sensors) {
	//copy necessary info back from cpu array to GPU array first
	data_util::boolD2H(dev_current, h_current, end_idx - start_idx, start_idx, start_idx);
	data_util::doubleD2H(dev_diag, h_diag, end_idx - start_idx, start_idx, start_idx);
	data_util::doubleD2H(dev_diag_, h_diag_, end_idx - start_idx, start_idx, start_idx);
	dataManagerLogger.debug("Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from GPU arrays to CPU arrays", _dependency);
	for (int i = start_idx; i < end_idx; ++i) {
		//bring all sensor and measurable info into the object
		try {
			_sensors[i]->pointers_to_values();
		}
		catch (UMAException &e) {
			dataManagerLogger.error("Fatal error while doing copy_arrays_to_sensors", _dependency);
			throw UMAException("Fatal error in function copy_arrays_to_sensors", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
		}
	}
	dataManagerLogger.debug("Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from cpu arrays to sensor", _dependency);
	dataManagerLogger.info("Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied back from arrays", _dependency);
}

void DataManager::copy_sensors_to_arrays(const int start_idx, const int end_idx, const vector<Sensor*> &_sensors) {
	//copy necessary info from sensor object to cpu arrays
	for (int i = start_idx; i < end_idx; ++i) {
		try {
			_sensors[i]->copyAmperList(h_mask_amper);
			_sensors[i]->values_to_pointers();
		}
		catch (exception &e) {
			dataManagerLogger.error("Fatal error while doing copy_sensors_to_arrays", _dependency);
			throw UMAException("Fatal error in function copy_sensors_to_arrays", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
		}
	}
	dataManagerLogger.debug("Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from sensor to CPU arrays", _dependency);
	//copy data from cpu array to GPU array
	data_util::boolH2D(h_mask_amper, dev_mask_amper, 2 * (ind(end_idx, 0) - ind(start_idx, 0)), 2 * ind(start_idx, 0), 2 * ind(start_idx, 0));
	data_util::boolH2D(h_current, dev_current, end_idx - start_idx, start_idx, start_idx);
	data_util::doubleH2D(h_diag, dev_diag, end_idx - start_idx, start_idx, start_idx);
	data_util::doubleH2D(h_diag_, dev_diag_, end_idx - start_idx, start_idx, start_idx);
	dataManagerLogger.debug("Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from CPU arrays to GPU arrays", _dependency);
	dataManagerLogger.info("Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied to arrays", _dependency);
}

/*
--------------------Sensor Object And Array Data Transfer---------------------
*/

/*
--------------------SensorPair Object And Array Data Transfer---------------------
*/
void DataManager::copy_sensor_pairs_to_arrays(const int start_idx, const int end_idx, const vector<SensorPair*> &_sensor_pairs) {
	//copy data from sensor pair object to CPU arrays
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		try {
			_sensor_pairs[i]->values_to_pointers();
		}
		catch (exception &e) {
			dataManagerLogger.error("Fatal error while doing copy_sensor_pairs_to_arrays", _dependency);
			throw UMAException("Fatal error in function copy_sensor_pairs_to_arrays", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
		}
	}
	dataManagerLogger.debug("Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from sensor pairs to CPU arrays", _dependency);
	//copy data from CPU arrays to GPU arrays
	data_util::doubleH2D(h_weights, dev_weights, (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)), ind(2 * start_idx, 0), ind(2 * start_idx, 0));
	data_util::boolH2D(h_dirs, dev_dirs, (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)), ind(2 * start_idx, 0), ind(2 * start_idx, 0));
	data_util::doubleH2D(h_thresholds, dev_thresholds, (ind(end_idx, 0) - ind(start_idx, 0)), ind(start_idx, 0), ind(start_idx, 0));
	dataManagerLogger.debug("Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from CPU arrays to GPU arrays", _dependency);
	dataManagerLogger.info("Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied to arrays", _dependency);
}

void DataManager::copy_arrays_to_sensor_pairs(const int start_idx, const int end_idx, const vector<SensorPair*> &_sensor_pairs) {
	//copy data from GPU arrays to CPU arrays
	data_util::doubleD2H(dev_weights, h_weights, (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)), ind(2 * start_idx, 0), ind(2 * start_idx, 0));
	data_util::boolD2H(dev_dirs, h_dirs, (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)), ind(2 * start_idx, 0), ind(2 * start_idx, 0));
	data_util::doubleD2H(dev_thresholds, h_thresholds, (ind(end_idx, 0) - ind(start_idx, 0)), ind(start_idx, 0), ind(start_idx, 0));
	dataManagerLogger.debug("Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from GPU arrays to CPU arrays", _dependency);
	//copy data from CPU arrays to sensor pairs
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		//bring all sensor pair info into the object
		try {
			_sensor_pairs[i]->pointers_to_values();
		}
		catch (exception &e) {
			dataManagerLogger.error("Fatal error while doing copy_arrays_to_sensor_pairs", _dependency);
			throw UMAException("Fatal error in function copy_arrays_to_sensor_pairs", UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
		}
	}
	dataManagerLogger.debug("Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from CPU arrays to sensor pairs", _dependency);
	dataManagerLogger.info("Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied back from arrays", _dependency);
}

/*
------------------------------SET FUNCTIONS--------------------------------
*/

/*
Set the mask used in halucination, for testing purpose, usual workflow, mask is get from gen_mask
Input: mask vector, size of mask have to be the _measurable_size
*/
void DataManager::setMask(const vector<bool> &mask) {
	if (mask.size() != _measurable_size) {
		throw UMAException("Input mask size not matching measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	for (int i = 0; i < mask.size(); ++i) h_mask[i] = mask[i];
	data_util::boolH2D(h_mask, dev_mask, mask.size());
	dataManagerLogger.debug("Mask value set", _dependency);
}

/*
This function set observe signal from python side
Input: observe signal
*/
void DataManager::setObserve(const vector<bool> &observe) {//this is where data comes in in every frame
	if (observe.size() != _measurable_size) {
		throw UMAException("Input observe size not matching measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	data_util::boolH2H(h_observe, h_observe_, _measurable_size);
	for (int i = 0; i < observe.size(); ++i) {
		h_observe[i] = observe[i];
	}
	data_util::boolH2D(h_observe, dev_observe, _measurable_size);
	dataManagerLogger.debug("observe signal set", _dependency);
}

/*
The function to set the current value, mainly used for testing puropse
Input: current signal
*/
void DataManager::setCurrent(const vector<bool> &current) {//this is where data comes in in every frame
	if (current.size() != _measurable_size) {
		throw UMAException("Input current size not matching measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	for (int i = 0; i < current.size(); ++i) {
		h_current[i] = current[i];
	}
	data_util::boolH2D(h_current, dev_current, _measurable_size);
	dataManagerLogger.debug("current signal set for customized purpose", _dependency);
}

/*
The function to set the target value
Input: target signal
*/
void DataManager::setTarget(const vector<bool> &target) {
	if (target.size() != _measurable_size) {
		throw UMAException("Input target size not matching measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	for (int i = 0; i < _measurable_size; ++i) {
		h_target[i] = target[i];
	}
	data_util::boolH2D(h_target, dev_target, _measurable_size);
	dataManagerLogger.debug("target signal set", _dependency);
}

/*
set Signal
Input: list of signal in bool, input size has to be measurable size
*/
void DataManager::setSignal(const vector<bool> &signal) {
	if (signal.size() != _measurable_size) {
		throw UMAException("The input signal size is not matching measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	for (int i = 0; i < _measurable_size; ++i) {
		h_signal[i] = signal[i];
	}
	data_util::boolH2D(h_signal, dev_signal, _measurable_size);
	dataManagerLogger.debug("signal set", _dependency);
}

/*
set Signals
Input: 2d vector of signals, first dimension should not exceed sensor size, second one must be measurable size
*/
void DataManager::setSignals(const vector<vector<bool> > &signals) {
	int sig_count = signals.size();
	if (sig_count > _measurable_size_max) {
		throw UMAException("The input sensor size is larger than current sensor size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	for (int i = 0; i < sig_count; ++i) {
		if (signals[i].size() != _measurable_size) {
			throw UMAException("The " + to_string(i) + "th input string size is not matching measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
		}
		for (int j = 0; j < signals[i].size(); ++j) {
			h_signals[i * _measurable_size + j] = signals[i][j];
		}
	}
	data_util::boolH2D(h_signals, dev_signals, sig_count * _measurable_size);
	dataManagerLogger.debug(to_string(sig_count) + " signals set", _dependency);
}

/*
set load
Input: list of load in bool, list size has to be measurable size
*/
void DataManager::setLoad(const vector<bool> &load) {
	if (load.size() != _measurable_size) {
		throw UMAException("The input load size is not matching the measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	for (int i = 0; i < _measurable_size; ++i) {
		h_load[i] = load[i];
	}
	data_util::boolH2D(h_load, dev_load, _measurable_size);
	dataManagerLogger.debug("load set", _dependency);
}

void DataManager::setDists(const vector<vector<int> > &dists) {
	if (dists.size() != _sensor_size) {
		throw UMAException("The input dists size is not matching the measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	for (int i = 0; i < dists.size(); ++i) {
		for (int j = 0; j < dists[0].size(); ++j) h_dists[i * _sensor_size + j] = dists[i][j];
	}
	data_util::intH2D(h_dists, dev_dists, _sensor_size * _sensor_size);
	dataManagerLogger.debug("dists set", _dependency);
}

/*
set signals with load
have to make sure load is set before calling this function
input: 2d signals vector
*/
void DataManager::setLSignals(const vector<vector<bool> > &signals) {
	int sig_count = signals.size();
	if (sig_count > _measurable_size_max) {
		throw UMAException("The input sensor size is larger than current sensor size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	for (int i = 0; i < sig_count; ++i) {
		if (signals[i].size() != _measurable_size) {
			throw UMAException("The " + to_string(i) + "th input string size is not matching measurable size!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
		}
		for (int j = 0; j < signals[i].size(); ++j) {
			h_lsignals[i * _measurable_size + j] = signals[i][j];
		}
	}
	data_util::boolH2D(h_lsignals, dev_lsignals, sig_count * _measurable_size);
	for (int i = 0; i < sig_count; ++i) {
		kernel_util::disjunction(dev_lsignals + i * _measurable_size, dev_load, _measurable_size);
	}
	dataManagerLogger.debug("loaded signals set", _dependency);
}

//------------------------------SET FUNCTIONS--------------------------------
//###########################################################################

//#########################################################################
//------------------------------GET FUNCTIONS--------------------------------

/*
Get signals
Output: 2d format of signals
*/
const vector<vector<bool> > DataManager::getSignals(int sig_count) {
	vector<vector<bool> > results;
	data_util::boolD2H(dev_signals, h_signals, sig_count * _measurable_size);
	for (int i = 0; i < sig_count; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_signals[i * _measurable_size + j]);
		results.push_back(tmp);
	}
	dataManagerLogger.debug(to_string(sig_count) + " signals get", _dependency);
	return results;
}

/*
Get loaded signals
Output: 2d format of loaded signals
*/
const vector<vector<bool> > DataManager::getLSignals(int sig_count) {
	vector<vector<bool> > results;
	data_util::boolD2H(dev_lsignals, h_lsignals, sig_count * _measurable_size);
	for (int i = 0; i < sig_count; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_lsignals[i * _measurable_size + j]);
		results.push_back(tmp);
	}
	dataManagerLogger.debug(to_string(sig_count) + " loaded signals get", _dependency);
	return results;
}

/*
Get Npdir masks
Output: 2d format of npdirs
*/
const vector<vector<bool> > DataManager::getNpdirMasks() {
	vector<vector<bool> > results;
	data_util::boolD2H(dev_npdir_mask, h_npdir_mask, _sensor_size * _measurable_size);
	for (int i = 0; i < _sensor_size; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_npdir_mask[i * _measurable_size + j]);
		results.push_back(tmp);
	}
	dataManagerLogger.debug("npdir mask get", _dependency);
	return results;
}

/*
get the current(observe value through propagation) value from GPU
Output: current vector
*/
const vector<bool> DataManager::getCurrent() {
	vector<bool> result;
	data_util::boolD2H(dev_current, h_current, _measurable_size);
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_current[i]);
	}
	dataManagerLogger.debug("current signal get", _dependency);
	return result;
}

/*
get the prediction(next iteration prediction) value from GPU
Output: prediction vector
*/
const vector<bool> DataManager::getPrediction() {
	vector<bool> result;
	//no corresponding dev variable to copy from, should be copied after the halucinate
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_prediction[i]);
	}
	dataManagerLogger.debug("prediction signal get", _dependency);
	return result;
}

/*
The function is getting the current target
Output: target vector
*/
const vector<bool> DataManager::getTarget() {
	vector<bool> result;
	data_util::boolD2H(dev_target, h_target, _measurable_size);
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_target[i]);
	}
	dataManagerLogger.debug("target signal get", _dependency);
	return result;
}

/*
The function is getting weight matrix in 2-dimensional form
Output: weight matrix in 2d format
*/
const vector<vector<double> > DataManager::getWeight2D() {
	data_util::doubleD2H(dev_weights, h_weights, _measurable2d_size);
	vector<vector<double> > result;
	int n = 0;
	for (int i = 0; i < _measurable_size; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_weights[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("weight matrix 2d get", _dependency);
	return result;
}

/*
The function is getting the dir matrix in 2-dimensional form
Output: dir matrix in 2d format
*/
const vector<vector<bool> > DataManager::getDir2D() {
	data_util::boolD2H(dev_dirs, h_dirs, _measurable2d_size);
	vector<vector<bool> > result;
	int n = 0;
	for (int i = 0; i < _measurable_size; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_dirs[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("dir matrix 2d get", _dependency);
	return result;
}

/*
The function is getting the n power of dir matrix in 2-dimensional form
Output: npdir matrix in 2d format
*/
const vector<vector<bool> > DataManager::getNPDir2D() {
	data_util::boolD2H(dev_npdirs, h_npdirs, _npdir_size);
	vector<vector<bool> > result;
	int n = 0;
	for (int i = 0; i < _measurable_size; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= i + (i % 2 == 0); ++j)
			tmp.push_back(h_npdirs[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("npdir matrix 2d get", _dependency);
	return result;
}

/*
The function is getting the threshold matrix in 2-dimensional form
Output: threshold matrix in 2d format
*/
const vector<vector<double> > DataManager::getThreshold2D() {
	data_util::doubleD2H(dev_thresholds, h_thresholds, _sensor2d_size);
	vector<vector<double> > result;
	int n = 0;
	for (int i = 0; i < _sensor_size; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_thresholds[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("threshold matrix 2d get", _dependency);
	return result;
}

/*
The function is getting the mask amper in 2-dimension form
Output: mask amper in 2d format
*/
const vector<vector<bool> > DataManager::getMask_amper2D() {
	vector<vector<bool> > result;
	data_util::boolD2H(dev_mask_amper, h_mask_amper, _mask_amper_size);
	int n = 0;
	for (int i = 0; i < _sensor_size; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= 2 * i + 1; ++j)
			tmp.push_back(h_mask_amper[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("mask amper 2d get", _dependency);
	return result;
}

/*
The function is getting mask amper as an list
Output: mask amper in 1d format
*/
const vector<bool> DataManager::getMask_amper() {
	vector<vector<bool> > tmp = getMask_amper2D();
	vector<bool> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	dataManagerLogger.debug("mask amper 1d get", _dependency);
	return results;
}

/*
The function is getting the weight matrix as a list
Output: weight matrix in 1d format
*/
const vector<double> DataManager::getWeight() {
	vector<vector<double > > tmp = getWeight2D();
	vector<double> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	dataManagerLogger.debug("weight matrix 1d get", _dependency);
	return results;
}

/*
The function is getting the dir matrix as a list
Output: dir matrix in 1d format
*/
const vector<bool> DataManager::getDir() {
	vector<vector<bool> > tmp = this->getDir2D();
	vector<bool> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	dataManagerLogger.debug("dir matrix 1d get", _dependency);
	return results;
}

/*
The function is getting the n power of dir matrix as a list
Output: npdir 1d format
*/
const vector<bool> DataManager::getNPDir() {
	vector<vector<bool> > tmp = this->getNPDir2D();
	vector<bool> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	dataManagerLogger.debug("npdir matrix 1d get", _dependency);
	return results;
}

/*
The function is getting the threshold matrix as a list
Output: threshold 1d format
*/
const vector<double> DataManager::getThresholds() {
	vector<vector<double> > tmp = getThreshold2D();
	vector<double> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	dataManagerLogger.debug("threshold matrix 1d get", _dependency);
	return results;
}

/*
This function is getting the diagonal value of the weight matrix
Output: diag value
*/
const vector<double> DataManager::getDiag() {
	data_util::doubleD2H(dev_diag, h_diag, _measurable_size);
	vector<double> result;
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_diag[i]);
	}
	dataManagerLogger.debug("diag value get", _dependency);
	return result;
}

/*
This function is getting the diagonal value of the weight matrix of last iteration
Output: old diag value
*/
const vector<double> DataManager::getDiagOld() {
	data_util::doubleD2H(dev_diag_, h_diag_, _measurable_size);
	vector<double> result;
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_diag_[i]);
	}
	dataManagerLogger.debug("old diag value get", _dependency);
	return result;
}

/*
This function is getting the current mask value used in halucination
Output: mask signal
*/
const vector<bool> DataManager::getMask() {
	vector<bool> result;
	data_util::boolD2H(dev_mask, h_mask, _measurable_size);
	for (int i = 0; i < this->_measurable_size; ++i) result.push_back(h_mask[i]);
	dataManagerLogger.debug("mask signal get", _dependency);
	return result;
}

/*
This function is getting the obersev matrix
Output: observe signal
*/
const vector<bool> DataManager::getObserve() {
	vector<bool> result;
	data_util::boolD2H(dev_observe, h_observe, _measurable_size);
	for (int i = 0; i < this->_measurable_size; ++i) result.push_back(h_observe[i]);
	dataManagerLogger.debug("observe signal get", _dependency);
	return result;
}

/*
This function get the load signal
Output: load signal
*/
const vector<bool> DataManager::getLoad() {
	vector<bool> result;
	data_util::boolD2H(dev_load, h_load, _measurable_size);
	for (int i = 0; i < _measurable_size; ++i) result.push_back(h_load[i]);
	dataManagerLogger.debug("load signal get", _dependency);
	return result;
}

/*
The function is getting the up signal
Output: up signal
*/
const vector<bool> DataManager::getUp() {
	vector<bool> result;
	//up have no corresponding dev variable, when doing up_GPU, the value should be stored and copied back to host
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_up[i]);
	}
	dataManagerLogger.debug("up signal get", _dependency);
	return result;
}

/*
The function is getting the down matrix
Output: down signal
*/
const vector<bool> DataManager::getDown() {
	vector<bool> result;
	//down have no corresponding dev variable, when doing up_GPU, the value should be stored and copied back to host
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_down[i]);
	}
	dataManagerLogger.debug("down signal get", _dependency);
	return result;
}

const vector<bool> DataManager::getNegligible() {
	vector<bool> result;
	data_util::boolD2H(dev_negligible, h_negligible, _measurable_size);
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_negligible[i]);
	}
	return result;
}

/*
The function is getting the union root in union_find
Output: the union root vector
*/
const vector<int> DataManager::getUnionRoot() {
	data_util::intD2H(dev_union_root, h_union_root, _sensor_size);
	vector<int> result;
	for (int i = 0; i < _sensor_size; ++i) {
		result.push_back(h_union_root[i]);
	}
	return result;
}

/*
------------------------------GET FUNCTIONS--------------------------------
*/

/*
This function returns the data manager related sizes
Output: a map from string->int, indicating all size
*/
const std::map<string, int> DataManager::getSizeInfo() {
	std::map<string, int> s;

	s["_sensor_size"] = _sensor_size;
	s["_sensor_size_max"] = _sensor_size_max;
	s["_sensor2d_size"] = _sensor2d_size;
	s["_sensor2d_size_max"] = _sensor2d_size_max;
	s["_measurable_size"] = _measurable_size;
	s["_measurable_size_max"] = _measurable_size_max;
	s["_measurable2d_size"] = _measurable2d_size;
	s["_measurable2d_size_max"] = _measurable2d_size_max;
	s["_mask_amper_size"] = _mask_amper_size;
	s["_mask_amper_size_max"] = _mask_amper_size_max;
	s["_npdir_size"] = _npdir_size;
	s["_npdir_size_max"] = _npdir_size_max;

	return s;
}

int *DataManager::_dvar_i(int name) {
	switch (name) {
	case DISTS: return dev_dists;
	case UNION_ROOT: return dev_union_root;
	}
}

double *DataManager::_dvar_d(int name) {
	switch (name) {
	case WEIGHTS: return dev_weights;
	case THRESHOLDS: return dev_thresholds;
	case DIAG: return dev_diag;
	case OLD_DIAG: return dev_diag_;
	}
}

float *DataManager::_dvar_f(int name) {
	switch (name) {
	case RES: return dev_res;
	}
}

bool *DataManager::_dvar_b(int name) {
	switch (name) {
	case DIRS: return dev_dirs;
	case OBSERVE: return dev_observe;
	case NPDIRS: return dev_npdirs;
	case SIGNALS: return dev_signals;
	case LSIGNALS: return dev_lsignals;
	case LOAD: return dev_load;
	case SIGNAL: return dev_signal;
	case CURRENT: return dev_current;
	case MASK: return dev_mask;
	case MASK_AMPER: return dev_mask_amper;
	case NPDIR_MASK: return dev_npdir_mask;
	case TARGET: return dev_target;
	case PREDICTION: return dev_prediction;
	case NEGLIGIBLE: return dev_negligible;
	}
}


int *DataManager::_hvar_i(int name) {
	switch (name) {
	case DISTS: return h_dists;
	case UNION_ROOT: return h_union_root;
	}
}

double *DataManager::_hvar_d(int name) {
	switch (name) {
	case WEIGHTS: return h_weights;
	case THRESHOLDS: return h_thresholds;
	case DIAG: return h_diag;
	case OLD_DIAG: return h_diag_;
	}
}

float *DataManager::_hvar_f(int name) {
	switch (name) {
	case RES: return h_res;
	}
}

bool *DataManager::_hvar_b(int name) {
	switch (name) {
	case DIRS: return h_dirs;
	case OBSERVE: return h_observe;
	case NPDIRS: return h_npdirs;
	case SIGNALS: return h_signals;
	case LOAD: return h_load;
	case SIGNAL: return h_signal;
	case CURRENT: return h_current;
	case MASK: return h_mask;
	case MASK_AMPER: return h_mask_amper;
	case PREDICTION: return h_prediction;
	case NPDIR_MASK: return h_npdir_mask;
	case TARGET: return h_target;
	case NEGLIGIBLE: return h_negligible;
	}
}