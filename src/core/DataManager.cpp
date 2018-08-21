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
DataManager::DataManager(UMACoreObject *parent): UMACoreObject("dataManager", UMA_OBJECT::DATA_MANAGER, parent) {
	_memoryExpansion = stod(World::coreInfo["DataManager"]["memory_expansion"]);

	initPointers();
	setSize(0);
	dataManagerLogger.info("Setting the memory expansion rate to " + to_string(_memoryExpansion));
}

DataManager::~DataManager() {
}

/*
init all pointers, to be NULL
*/
void DataManager::initPointers() {
	h_dirs = NULL;
	h_weights = NULL;
	h_thresholds = NULL;
	h_mask_amper = NULL;
	h_npdirs = NULL;
	h_observe = NULL;
	h_observe_ = NULL;
	h_load = NULL;
	h_current = NULL;
	h_current_ = NULL;
	h_mask = NULL;
	h_target = NULL;
	h_prediction = NULL;
	h_diag = NULL;
	h_diag_ = NULL;
	h_npdir_mask = NULL;
	h_signals = NULL;
	h_bool_tmp = NULL;
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
	dev_load = NULL;
	dev_bool_tmp = NULL;
	dev_current = NULL;
	dev_current_ = NULL;
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
	dev_sum = NULL;
	
	dev_dec_tmp1 = NULL;
	dev_dec_tmp2 = NULL;

	dataManagerLogger.debug("Setting all pointers to NULL");
}

/*
Remalloc the memory, based on the sensor size
Input: the sensor size
*/
void DataManager::reallocateMemory(double &total, int sensorSize) {
	dataManagerLogger.info("Starting reallocating memory");
	//free all memory first
	freeAllParameters();
	//then set the size to be new one
	setSize(sensorSize, true);

	try {
		//then generate all matrix again
		genWeight();
		genDirection();
		genThresholds();
		genMaskAmper();
		genNpDirection();
		genSignals();
		genNpdirMask();
		genDists();
		genOtherParameters();
	}
	catch (UMAException &e) {
		throw UMAInternalException("Fatal error in reallocate_memory when doing memory allocation", true, &dataManagerLogger);
	}

	try {
		//init other parameters
		initOtherParameter(total);
	}
	catch (UMAException &e) {
		throw UMAInternalException("Fatal error in reallocate_memory when doing parameters ini", true, &dataManagerLogger);
	}
	dataManagerLogger.info("Memory reallocated!");
}

/*
Set the size
Input: new sensor size and whether changing max of not
*/
void DataManager::setSize(int sensorSize, bool change_max) {
	_sensorSize = sensorSize;
	_attrSensorSize = 2 * _sensorSize;
	_sensor2dSize = _sensorSize * (_sensorSize + 1) / 2;
	_attrSensor2dSize = _attrSensorSize * (_attrSensorSize + 1) / 2;
	_maskAmperSize = _sensorSize * (_sensorSize + 1);
	_npdirSize = _attrSensor2dSize + _sensorSize;

	dataManagerLogger.info("Setting sensor size to " + to_string(_sensorSize));
	dataManagerLogger.debug("Setting attr_sensor size to " + to_string(_attrSensorSize));
	dataManagerLogger.debug("Setting sensor size 2D to " + to_string(_sensor2dSize));
	dataManagerLogger.debug("Setting attr_sensor size 2D to " + to_string(_attrSensor2dSize));
	dataManagerLogger.debug("Setting mask amper size to " + to_string(_maskAmperSize));
	dataManagerLogger.debug("Setting npdir size to " + to_string(_npdirSize));

	if (change_max) {
		_sensorSizeMax = (int)(_sensorSize * (1 + _memoryExpansion));
		_attrSensorSizeMax = 2 * _sensorSizeMax;
		_sensor2dSizeMax = _sensorSizeMax * (_sensorSizeMax + 1) / 2;
		_attrSensor2dSizeMax = _attrSensorSizeMax * (_attrSensorSizeMax + 1) / 2;
		_maskAmperSizeMax = _sensorSizeMax * (_sensorSizeMax + 1);
		_npdirSizeMax = _attrSensor2dSizeMax + _sensorSizeMax;

		dataManagerLogger.debug("Setting max sensor size to " + to_string(_sensorSizeMax));
		dataManagerLogger.debug("Setting max attr_sensor size to " + to_string(_attrSensorSizeMax));
		dataManagerLogger.debug("Setting max sensor size 2D to " + to_string(_sensor2dSizeMax));
		dataManagerLogger.debug("Setting max attr_sensor size 2D to " + to_string(_attrSensor2dSizeMax));
		dataManagerLogger.debug("Setting max mask amper size to " + to_string(_maskAmperSizeMax));
		dataManagerLogger.debug("Setting max npdir size to " + to_string(_npdirSizeMax));
	}
	else dataManagerLogger.info("All size max value remain the same");
}

/*
init other parameters
Input: total value
*/
void DataManager::initOtherParameter(double &total) {
	for (int i = 0; i < _attrSensorSizeMax; ++i) {
		h_observe[i] = false;
		h_observe_[i] = false;
		h_load[i] = false;
		h_bool_tmp[i] = false;
		h_mask[i] = false;
		h_current[i] = false;
		h_current_[i] = false;
		h_target[i] = false;
		h_prediction[i] = false;
		h_diag[i] = total / 2.0;
		h_diag_[i] = total / 2.0;
		h_negligible[i] = false;
	}
	*h_res = 0.0;
	for (int i = 0; i < _sensorSizeMax; ++i) {
		h_union_root[i] = 0;
	}

	data_util::dev_init(dev_observe, _attrSensorSizeMax);
	data_util::dev_init(dev_load, _attrSensorSizeMax);
	data_util::dev_init(dev_bool_tmp, _attrSensorSizeMax);
	data_util::dev_init(dev_mask, _attrSensorSizeMax);
	data_util::dev_init(dev_current, _attrSensorSizeMax);
	data_util::dev_init(dev_current_, _attrSensorSizeMax);
	data_util::dev_init(dev_target, _attrSensorSizeMax);
	data_util::dev_init(dev_prediction, _attrSensorSizeMax);
	data_util::dev_init(dev_negligible, _attrSensorSizeMax);
	data_util::dev_init(dev_res, 1);
	data_util::dev_init(dev_sum, _attrSensorSizeMax);
	data_util::dev_init(dev_union_root, _sensorSizeMax);

	data_util::dev_init(dev_dec_tmp1, _attrSensorSizeMax);
	data_util::dev_init(dev_dec_tmp2, _attrSensorSizeMax);

	uma_base::initDiag(dev_diag, dev_diag_, total, total, _attrSensorSizeMax);
}


/*
This function free data together
Deprecated in new version
Input: None
Output: None
*/
void DataManager::freeAllParameters() {//free data in case of memory leak
	dataManagerLogger.info("Starting releasing memory");
	try {
		delete[] h_dirs;
		delete[] h_weights;
		delete[] h_thresholds;
		delete[] h_mask_amper;
		delete[] h_observe;
		delete[] h_observe_;
		delete[] h_load;
		delete[] h_mask;
		delete[] h_current;
		delete[] h_current_;
		delete[] h_target;
		delete[] h_negligible;
		delete[] h_diag;
		delete[] h_diag_;
		delete[] h_prediction;
		delete[] h_npdirs;
		delete[] h_npdir_mask;
		delete[] h_signals;
		delete[] h_bool_tmp;
		delete[] h_dists;
		delete[] h_union_root;
		delete h_res;
	}
	catch (UMAException &e) {
		throw UMAInternalException("Fatal error in free_all_parameters, when doing cpu array release", true, &dataManagerLogger);
	}

	try {
		data_util::dev_free(dev_dirs);
		data_util::dev_free(dev_weights);
		data_util::dev_free(dev_thresholds);
		data_util::dev_free(dev_mask_amper);

		data_util::dev_free(dev_mask);
		data_util::dev_free(dev_current);
		data_util::dev_free(dev_current_);
		data_util::dev_free(dev_target);

		data_util::dev_free(dev_observe);
		data_util::dev_free(dev_load);
		data_util::dev_free(dev_bool_tmp);

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
		data_util::dev_free(dev_sum);

		data_util::dev_free(dev_dec_tmp1);
		data_util::dev_free(dev_dec_tmp2);
	}
	catch (UMAException &e) {
		throw UMAInternalException("Fatal error in free_all_parameters, when doing gpu array release", true, &dataManagerLogger);
	}
	dataManagerLogger.info("All memory released");
}


/*
This function is generating dir matrix memory both on host and device
*/
void DataManager::genDirection() {
	//malloc the space
	h_dirs = new bool[_attrSensor2dSizeMax];
	data_util::dev_bool(dev_dirs, _attrSensor2dSizeMax);

	//fill in all 0
	memset(h_dirs, 0, _attrSensor2dSizeMax * sizeof(bool));
	data_util::dev_init(dev_dirs, _attrSensor2dSizeMax);

	dataManagerLogger.debug("Dir matrix generated with size " + to_string(_attrSensor2dSizeMax));
}

/*
This function is generating weight matrix memory both on host and device
*/
void DataManager::genWeight() {
	//malloc the space
	h_weights = new double[_attrSensor2dSizeMax];
	data_util::dev_double(dev_weights, _attrSensor2dSizeMax);

	//fill in all 0
	memset(h_weights, 0, _attrSensor2dSizeMax * sizeof(double));
	data_util::dev_init(dev_weights, _attrSensor2dSizeMax);

	dataManagerLogger.debug("Weight matrix generated with size " + to_string(_attrSensor2dSizeMax));
}

/*
This function is generating threshold matrix memory both on host and device
*/
void DataManager::genThresholds() {
	//malloc the space
	h_thresholds = new double[_sensor2dSizeMax];
	data_util::dev_double(dev_thresholds, _sensor2dSizeMax);

	//fill in all 0
	memset(h_thresholds, 0, _sensor2dSizeMax * sizeof(double));
	data_util::dev_init(dev_thresholds, _sensor2dSizeMax);

	dataManagerLogger.debug("Threshold matrix generated with the size " + to_string(_sensor2dSizeMax));
}

/*
This function is generating mask amper matrix memory both on host and device
*/
void DataManager::genMaskAmper() {
	//malloc the space
	h_mask_amper = new bool[_maskAmperSizeMax];
	data_util::dev_bool(dev_mask_amper, _maskAmperSizeMax);

	//fill in all 0
	memset(h_mask_amper, 0, _maskAmperSizeMax * sizeof(bool));
	data_util::dev_init(dev_mask_amper, _maskAmperSizeMax);

	dataManagerLogger.debug("Mask amper matrix generated with size " + to_string(_maskAmperSizeMax));
}

/*
This function is generating n power of dir matrix both on host and device
*/
void DataManager::genNpDirection() {
	//malloc the space
	h_npdirs = new bool[_npdirSizeMax];
	data_util::dev_bool(dev_npdirs, _npdirSizeMax);

	//fill in all 0
	memset(h_npdirs, 0, _npdirSizeMax * sizeof(bool));
	data_util::dev_init(dev_npdirs, _npdirSizeMax);

	dataManagerLogger.debug("NPDir matrix generated with size " + to_string(_npdirSizeMax));
}

void DataManager::genSignals() {
	//malloc the space
	h_signals = new bool[_attrSensorSizeMax * _attrSensorSizeMax];
	h_lsignals = new bool[_attrSensorSizeMax * _attrSensorSizeMax];
	data_util::dev_bool(dev_signals, _attrSensorSizeMax * _attrSensorSizeMax);
	data_util::dev_bool(dev_lsignals, _attrSensorSizeMax * _attrSensorSizeMax);

	//init with all false
	memset(h_signals, 0, _attrSensorSizeMax * _attrSensorSizeMax * sizeof(bool));
	memset(h_lsignals, 0, _attrSensorSizeMax * _attrSensorSizeMax * sizeof(bool));
	data_util::dev_init(dev_signals, _attrSensorSizeMax * _attrSensorSizeMax);
	data_util::dev_init(dev_lsignals, _attrSensorSizeMax * _attrSensorSizeMax);

	dataManagerLogger.debug(to_string(_attrSensorSizeMax) + " num of signals with length " + to_string(_attrSensorSizeMax) + " are generated, total size " + to_string(_attrSensorSizeMax * _attrSensorSizeMax));
	dataManagerLogger.debug(to_string(_attrSensorSizeMax) + " num of loaded signals with length " + to_string(_attrSensorSizeMax) + " are generated, total size " + to_string(_attrSensorSizeMax * _attrSensorSizeMax));
}

void DataManager::genNpdirMask() {
	//malloc the space
	h_npdir_mask = new bool[_sensorSizeMax * _attrSensorSizeMax];
	data_util::dev_bool(dev_npdir_mask, _sensorSizeMax * _attrSensorSizeMax);

	//init with all false
	memset(h_npdir_mask, 0, _sensorSizeMax * _attrSensorSizeMax * sizeof(bool));
	data_util::dev_init(dev_npdir_mask, _sensorSizeMax * _attrSensorSizeMax);

	dataManagerLogger.debug(to_string(_sensorSizeMax) + " num of npdir mask with length " + to_string(_attrSensorSizeMax) + " are generated, total size " + to_string(_sensorSizeMax * _attrSensorSizeMax));
}

void DataManager::genDists() {
	//malloc the space
	h_dists = new int[_attrSensorSizeMax * _attrSensorSizeMax];
	data_util::dev_int(dev_dists, _attrSensorSizeMax * _attrSensorSizeMax);

	//init with all 0
	memset(h_dists, 0, _attrSensorSizeMax * _attrSensorSizeMax * sizeof(int));
	data_util::dev_init(dev_dists, _attrSensorSizeMax * _attrSensorSizeMax);

	dataManagerLogger.debug(to_string(_attrSensorSizeMax) + "*" + to_string(_attrSensorSizeMax) + "=" + to_string(_attrSensorSizeMax * _attrSensorSizeMax) + " num of space allocated for dists, used for block GPU");
}

/*
This function generate other parameter
*/
void DataManager::genOtherParameters() {
	h_observe = new bool[_attrSensorSizeMax];
	h_observe_ = new bool[_attrSensorSizeMax];
	h_bool_tmp = new bool[_attrSensorSizeMax];
	h_load = new bool[_attrSensorSizeMax];
	h_mask = new bool[_attrSensorSizeMax];
	h_current = new bool[_attrSensorSizeMax];
	h_current_ = new bool[_attrSensorSizeMax];
	h_target = new bool[_attrSensorSizeMax];
	h_negligible = new bool[_attrSensorSizeMax];
	h_diag = new double[_attrSensorSizeMax];
	h_diag_ = new double[_attrSensorSizeMax];
	h_prediction = new bool[_attrSensorSizeMax];
	h_union_root = new int[_sensorSizeMax];
	h_res = new float;

	data_util::dev_bool(dev_observe, _attrSensorSizeMax);
	data_util::dev_bool(dev_load, _attrSensorSizeMax);
	data_util::dev_bool(dev_bool_tmp, _attrSensorSizeMax);
	data_util::dev_bool(dev_mask, _attrSensorSizeMax);
	data_util::dev_bool(dev_current, _attrSensorSizeMax);
	data_util::dev_bool(dev_current_, _attrSensorSizeMax);
	data_util::dev_bool(dev_target, _attrSensorSizeMax);
	data_util::dev_bool(dev_prediction, _attrSensorSizeMax);
	data_util::dev_bool(dev_negligible, _attrSensorSizeMax);

	data_util::dev_double(dev_diag, _attrSensorSizeMax);
	data_util::dev_double(dev_diag_, _attrSensorSizeMax);
	data_util::dev_int(dev_union_root, _sensorSizeMax);
	data_util::dev_float(dev_res, 1);
	data_util::dev_double(dev_sum, _attrSensorSizeMax);

	data_util::dev_bool(dev_dec_tmp1, _attrSensorSizeMax);
	data_util::dev_bool(dev_dec_tmp2, _attrSensorSizeMax);

	dataManagerLogger.debug("Other parameter generated, with size " + to_string(_attrSensorSizeMax));
}

void DataManager::createSensorsToArraysIndex(const int startIdx, const int endIdx, const vector<Sensor*> &sensors) {
	//create idx for diag and current
	if (startIdx < 0 || endIdx > _sensorSize) {
		throw UMAInvalidArgsException("The input index range of sensor is illegal!", false, &dataManagerLogger);
	}
	for (int i = startIdx; i < endIdx; ++i) {
		try {
			sensors[i]->setAttrSensorDiagPointers(h_diag, h_diag_);
			sensors[i]->setAttrSensorObservePointers(h_observe, h_observe_);
			sensors[i]->setAttrSensorCurrentPointers(h_current, h_current_);
			sensors[i]->setAttrSensorTargetPointers(h_target);
			sensors[i]->setAttrSensorPredictionPointers(h_prediction);
		}
		catch (exception &e) {
			throw UMAInternalException("Fatal error happen in createSensorsToArraysIndex", true, &dataManagerLogger);
		}
	}
	dataManagerLogger.info("Sensor from idx " + to_string(startIdx) + " to " + to_string(endIdx) + " have created idx to arrays");
}

void DataManager::createSensorPairsToArraysIndex(const int startIdx, const int endIdx, const vector<SensorPair*> &sensorPairs) {
	if (startIdx < 0 || endIdx > _sensorSize) {
		throw UMAInvalidArgsException("The input index range of sensor is illegal!", false, &dataManagerLogger);
	}
	for (int i = ind(startIdx, 0); i < ind(endIdx, 0); ++i) {
		try {
			sensorPairs[i]->setAllPointers(h_weights, h_dirs, h_thresholds);
		}
		catch (exception &e) {
			throw UMAInternalException("Fatal error happen in createSensorPairsToArraysIndex", true, &dataManagerLogger);
		}
	}
	dataManagerLogger.info("Sensor pairs from idx " + to_string(ind(startIdx, 0)) + " to " + to_string(ind(endIdx, 0)) + " have created idx to arrays");
}

/*
--------------------Sensor Object And Array Data Transfer---------------------
*/
void DataManager::copyArraysToSensors(const int startIdx, const int endIdx, const vector<Sensor*> &_sensors) {
	//copy necessary info back from cpu array to GPU array first
	data_util::doubleD2H(dev_diag, h_diag, 2 * (endIdx - startIdx), 2 * startIdx, 2 * startIdx);
	data_util::doubleD2H(dev_diag_, h_diag_, 2 * (endIdx - startIdx), 2 * startIdx, 2 * startIdx);
	data_util::boolD2H(dev_target, h_target, 2 * (endIdx - startIdx), 2 * startIdx, 2 * startIdx);
	dataManagerLogger.debug("Sensor data from idx " + to_string(startIdx) + " to " + to_string(endIdx) + " are copied from GPU arrays to CPU arrays");
	for (int i = startIdx; i < endIdx; ++i) {
		//bring all sensor and attr_sensor info into the object
		try {
			_sensors[i]->pointersToValues();
		}
		catch (UMAException &e) {
			throw UMAInternalException("Fatal error in function copy_arrays_to_sensors", true, &dataManagerLogger);
		}
	}
	dataManagerLogger.debug("Sensor data from idx " + to_string(startIdx) + " to " + to_string(endIdx) + " are copied from cpu arrays to sensor");
	dataManagerLogger.info("Sensor data from idx " + to_string(startIdx) + " to " + to_string(endIdx) + " are copied back from arrays");
}

void DataManager::copySensorsToArrays(const int startIdx, const int endIdx, const vector<Sensor*> &_sensors) {
	//copy necessary info from sensor object to cpu arrays
	for (int i = startIdx; i < endIdx; ++i) {
		try {
			_sensors[i]->copyAmperList(h_mask_amper);
			_sensors[i]->valuesToPointers();
		}
		catch (exception &e) {
			throw UMAInternalException("Fatal error in function copySensorsToArrays", true, &dataManagerLogger);
		}
	}
	dataManagerLogger.debug("Sensor data from idx " + to_string(startIdx) + " to " + to_string(endIdx) + " are copied from sensor to CPU arrays");
	//copy data from cpu array to GPU array
	data_util::boolH2D(h_mask_amper, dev_mask_amper, 2 * (ind(endIdx, 0) - ind(startIdx, 0)), 2 * ind(startIdx, 0), 2 * ind(startIdx, 0));
	data_util::doubleH2D(h_diag, dev_diag, 2 * (endIdx - startIdx), 2 * startIdx, 2 * startIdx);
	data_util::doubleH2D(h_diag_, dev_diag_, 2 * (endIdx - startIdx), 2 * startIdx, 2 * startIdx);
	data_util::boolH2D(h_target, dev_target, 2 * (endIdx - startIdx), 2 * startIdx, 2 * startIdx);
	dataManagerLogger.debug("Sensor data from idx " + to_string(startIdx) + " to " + to_string(endIdx) + " are copied from CPU arrays to GPU arrays");
	dataManagerLogger.info("Sensor data from idx " + to_string(startIdx) + " to " + to_string(endIdx) + " are copied to arrays");
}

/*
--------------------Sensor Object And Array Data Transfer---------------------
*/

/*
--------------------SensorPair Object And Array Data Transfer---------------------
*/
void DataManager::copySensorPairsToArrays(const int startIdx, const int endIdx, const vector<SensorPair*> &_sensorPairs) {
	//copy data from sensor pair object to CPU arrays
	for (int i = ind(startIdx, 0); i < ind(endIdx, 0); ++i) {
		try {
			_sensorPairs[i]->valuesToPointers();
		}
		catch (exception &e) {
			throw UMAInternalException("Fatal error in function copySensorPairsToArrays", true, &dataManagerLogger);
		}
	}
	dataManagerLogger.debug("Sensor pairs data from idx " + to_string(ind(startIdx, 0)) + " to " + to_string(ind(endIdx, 0)) + " are copied from sensor pairs to CPU arrays");
	//copy data from CPU arrays to GPU arrays
	data_util::doubleH2D(h_weights, dev_weights, (ind(2 * endIdx, 0) - ind(2 * startIdx, 0)), ind(2 * startIdx, 0), ind(2 * startIdx, 0));
	data_util::boolH2D(h_dirs, dev_dirs, (ind(2 * endIdx, 0) - ind(2 * startIdx, 0)), ind(2 * startIdx, 0), ind(2 * startIdx, 0));
	data_util::doubleH2D(h_thresholds, dev_thresholds, (ind(endIdx, 0) - ind(startIdx, 0)), ind(startIdx, 0), ind(startIdx, 0));
	dataManagerLogger.debug("Sensor pairs data from idx " + to_string(ind(startIdx, 0)) + " to " + to_string(ind(endIdx, 0)) + " are copied from CPU arrays to GPU arrays");
	dataManagerLogger.info("Sensor pairs data from idx " + to_string(ind(startIdx, 0)) + " to " + to_string(ind(endIdx, 0)) + " are copied to arrays");
}

void DataManager::copyArraysToSensorPairs(const int startIdx, const int endIdx, const vector<SensorPair*> &_sensorPairs) {
	//copy data from GPU arrays to CPU arrays
	data_util::doubleD2H(dev_weights, h_weights, (ind(2 * endIdx, 0) - ind(2 * startIdx, 0)), ind(2 * startIdx, 0), ind(2 * startIdx, 0));
	data_util::boolD2H(dev_dirs, h_dirs, (ind(2 * endIdx, 0) - ind(2 * startIdx, 0)), ind(2 * startIdx, 0), ind(2 * startIdx, 0));
	data_util::doubleD2H(dev_thresholds, h_thresholds, (ind(endIdx, 0) - ind(startIdx, 0)), ind(startIdx, 0), ind(startIdx, 0));
	dataManagerLogger.debug("Sensor pairs data from idx " + to_string(ind(startIdx, 0)) + " to " + to_string(ind(endIdx, 0)) + " are copied from GPU arrays to CPU arrays");
	//copy data from CPU arrays to sensor pairs
	for (int i = ind(startIdx, 0); i < ind(endIdx, 0); ++i) {
		//bring all sensor pair info into the object
		try {
			_sensorPairs[i]->pointersToValues();
		}
		catch (exception &e) {
			throw UMAInternalException("Fatal error in function copyArraysToSensorPairs", true, &dataManagerLogger);
		}
	}
	dataManagerLogger.debug("Sensor pairs data from idx " + to_string(ind(startIdx, 0)) + " to " + to_string(ind(endIdx, 0)) + " are copied from CPU arrays to sensor pairs");
	dataManagerLogger.info("Sensor pairs data from idx " + to_string(ind(startIdx, 0)) + " to " + to_string(ind(endIdx, 0)) + " are copied back from arrays");
}

/*
------------------------------SET FUNCTIONS--------------------------------
*/

/*
Set the mask used in halucination, for testing purpose, usual workflow, mask is get from genMask
Input: mask vector, size of mask have to be the _attrSensorSize
*/
void DataManager::setMask(const vector<bool> &mask) {
	if (mask.size() != _attrSensorSize) {
		throw UMAInvalidArgsException("Input mask size not matching attr_sensor size!", false, &dataManagerLogger);
	}
	for (int i = 0; i < mask.size(); ++i) h_mask[i] = mask[i];
	data_util::boolH2D(h_mask, dev_mask, mask.size());
	dataManagerLogger.debug("Mask value set");
}

/*
This function set observe signal from python side
Input: observe signal
*/
void DataManager::setObserve(const vector<bool> &observe) {//this is where data comes in in every frame
	if (observe.size() != _attrSensorSize) {
		throw UMAInvalidArgsException("The input observe size is not the size of attr sensor size", false, &dataManagerLogger);
	}
	data_util::boolH2H(h_observe, h_observe_, _attrSensorSize);
	for (int i = 0; i < observe.size(); ++i) {
		h_observe[i] = observe[i];
	}
	data_util::boolH2D(h_observe, dev_observe, _attrSensorSize);
	dataManagerLogger.debug("observe signal set");
}

/*
The function to set the current value, mainly used for testing puropse
Input: current signal
*/
void DataManager::setCurrent(const vector<bool> &current) {//this is where data comes in in every frame
	if (current.size() != _attrSensorSize) {
		string s = "Input current size not matching attr_sensor size!";
		throw UMAInvalidArgsException("Input current size not matching attr_sensor size!", false, &dataManagerLogger);
	}
	for (int i = 0; i < current.size(); ++i) {
		h_current[i] = current[i];
	}
	data_util::boolH2D(h_current, dev_current, _attrSensorSize);
	dataManagerLogger.debug("current signal set for customized purpose");
}

/*
The function to set the old current value, mainly used for testing puropse
Input: current signal
*/
void DataManager::setOldCurrent(const vector<bool> &current) {//this is where data comes in in every frame
	if (current.size() != _attrSensorSize) {
		throw UMAInvalidArgsException("Input old current size not matching attr_sensor size!", false, &dataManagerLogger);
	}
	for (int i = 0; i < current.size(); ++i) {
		h_current_[i] = current[i];
	}
	data_util::boolH2D(h_current_, dev_current_, _attrSensorSize);
	dataManagerLogger.debug("old current signal set for customized purpose");
}

/*
The function to set the target value
Input: target signal
*/
void DataManager::setTarget(const vector<bool> &target) {
	if (target.size() != _attrSensorSize) {
		throw UMAInvalidArgsException("Input target size not matching attr_sensor size!", false, &dataManagerLogger);
	}
	for (int i = 0; i < _attrSensorSize; ++i) {
		h_target[i] = target[i];
	}
	data_util::boolH2D(h_target, dev_target, _attrSensorSize);
	dataManagerLogger.debug("target signal set");
}

/*
set Signals
Input: 2d vector of signals, first dimension should not exceed sensor size, second one must be attr_sensor size
*/
void DataManager::setSignals(const vector<vector<bool> > &signals) {
	int sigCount = signals.size();
	if (sigCount > _attrSensorSizeMax) {
		throw UMAInvalidArgsException("The input sensor size is larger than current sensor size!", false, &dataManagerLogger);
	}
	for (int i = 0; i < sigCount; ++i) {
		if (signals[i].size() != _attrSensorSize) {
			throw UMAInvalidArgsException("The " + to_string(i) + "th input string size is not matching attr_sensor size!", false, &dataManagerLogger);
		}
		for (int j = 0; j < signals[i].size(); ++j) {
			h_signals[i * _attrSensorSize + j] = signals[i][j];
		}
	}
	data_util::boolH2D(h_signals, dev_signals, sigCount * _attrSensorSize);
	dataManagerLogger.debug(to_string(sigCount) + " signals set");
}

/*
set load
Input: list of load in bool, list size has to be attr_sensor size
*/
void DataManager::setLoad(const vector<bool> &load) {
	if (load.size() != _attrSensorSize) {
		throw UMAInvalidArgsException("The input load size is not matching the attr_sensor size!", false, &dataManagerLogger);
	}
	for (int i = 0; i < _attrSensorSize; ++i) {
		h_load[i] = load[i];
	}
	data_util::boolH2D(h_load, dev_load, _attrSensorSize);
	dataManagerLogger.debug("load set");
}

void DataManager::setDists(const vector<vector<int> > &dists) {
	if (dists.size() > _sensorSize) {
		throw UMAInvalidArgsException("The input dists size is larger than the sensor size!", false, &dataManagerLogger);
	}
	for (int i = 0; i < dists.size(); ++i) {
		if (dists[i].size() != _sensorSize) {
			throw UMAInvalidArgsException("The " + to_string(i) + "th input dists size is larger than sensor size!", false, &dataManagerLogger);
		}
		for (int j = 0; j < dists[0].size(); ++j) h_dists[i * _sensorSize + j] = dists[i][j];
	}
	data_util::intH2D(h_dists, dev_dists, _sensorSize * _sensorSize);
	dataManagerLogger.debug("dists set");
}

/*
set signals with load
have to make sure load is set before calling this function
input: 2d signals vector
*/
void DataManager::setLSignals(const vector<vector<bool> > &signals) {
	int sigCount = signals.size();
	if (sigCount > _attrSensorSizeMax) {
		throw UMAInvalidArgsException("The input sensor size is larger than current sensor size!", false, &dataManagerLogger);
	}
	for (int i = 0; i < sigCount; ++i) {
		if (signals[i].size() != _attrSensorSize) {
			throw UMAInvalidArgsException("The " + to_string(i) + "th input string size is not matching attr_sensor size!", false, &dataManagerLogger);
		}
		for (int j = 0; j < signals[i].size(); ++j) {
			h_lsignals[i * _attrSensorSize + j] = signals[i][j];
		}
	}
	data_util::boolH2D(h_lsignals, dev_lsignals, sigCount * _attrSensorSize);
	for (int i = 0; i < sigCount; ++i) {
		kernel_util::disjunction(dev_lsignals + i * _attrSensorSize, dev_load, _attrSensorSize);
	}
	dataManagerLogger.debug("loaded signals set");
}

//------------------------------SET FUNCTIONS--------------------------------
//###########################################################################

//#########################################################################
//------------------------------GET FUNCTIONS--------------------------------

/*
Get signals
Output: 2d format of signals
*/
const vector<vector<bool> > DataManager::getSignals(int sigCount) {
	vector<vector<bool> > results;
	data_util::boolD2H(dev_signals, h_signals, sigCount * _attrSensorSize);
	for (int i = 0; i < sigCount; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _attrSensorSize; ++j) tmp.push_back(h_signals[i * _attrSensorSize + j]);
		results.push_back(tmp);
	}
	dataManagerLogger.debug(to_string(sigCount) + " signals get");
	return results;
}

/*
Get loaded signals
Output: 2d format of loaded signals
*/
const vector<vector<bool> > DataManager::getLSignals(int sigCount) {
	vector<vector<bool> > results;
	data_util::boolD2H(dev_lsignals, h_lsignals, sigCount * _attrSensorSize);
	for (int i = 0; i < sigCount; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _attrSensorSize; ++j) tmp.push_back(h_lsignals[i * _attrSensorSize + j]);
		results.push_back(tmp);
	}
	dataManagerLogger.debug(to_string(sigCount) + " loaded signals get");
	return results;
}

/*
Get Npdir masks
Output: 2d format of npdirs
*/
const vector<vector<bool> > DataManager::getNpdirMasks() {
	vector<vector<bool> > results;
	data_util::boolD2H(dev_npdir_mask, h_npdir_mask, _sensorSize * _attrSensorSize);
	for (int i = 0; i < _sensorSize; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _attrSensorSize; ++j) tmp.push_back(h_npdir_mask[i * _attrSensorSize + j]);
		results.push_back(tmp);
	}
	dataManagerLogger.debug("npdir mask get");
	return results;
}

/*
get the current(observe value through propagation) value from GPU
Output: current vector
*/
const vector<bool> DataManager::getCurrent() {
	vector<bool> result;
	data_util::boolD2H(dev_current, h_current, _attrSensorSize);
	for (int i = 0; i < _attrSensorSize; ++i) {
		result.push_back(h_current[i]);
	}
	dataManagerLogger.debug("current signal get");
	return result;
}

const vector<bool> DataManager::getOldCurrent() {
	vector<bool> result;
	data_util::boolD2H(dev_current_, h_current_, _attrSensorSize);
	for (int i = 0; i < _attrSensorSize; ++i) {
		result.push_back(h_current_[i]);
	}
	dataManagerLogger.debug("old current signal get");
	return result;
}

/*
get the prediction(next iteration prediction) value from GPU
Output: prediction vector
*/
const vector<bool> DataManager::getPrediction() {
	vector<bool> result;
	//no corresponding dev variable to copy from, should be copied after the halucinate
	for (int i = 0; i < _attrSensorSize; ++i) {
		result.push_back(h_prediction[i]);
	}
	dataManagerLogger.debug("prediction signal get");
	return result;
}

/*
The function is getting the current target
Output: target vector
*/
const vector<bool> DataManager::getTarget() {
	vector<bool> result;
	data_util::boolD2H(dev_target, h_target, _attrSensorSize);
	for (int i = 0; i < _attrSensorSize; ++i) {
		result.push_back(h_target[i]);
	}
	dataManagerLogger.debug("target signal get");
	return result;
}

/*
The function is getting weight matrix in 2-dimensional form
Output: weight matrix in 2d format
*/
const vector<vector<double> > DataManager::getWeight2D() {
	data_util::doubleD2H(dev_weights, h_weights, _attrSensor2dSize);
	vector<vector<double> > result;
	int n = 0;
	for (int i = 0; i < _attrSensorSize; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_weights[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("weight matrix 2d get");
	return result;
}

/*
The function is getting the dir matrix in 2-dimensional form
Output: dir matrix in 2d format
*/
const vector<vector<bool> > DataManager::getDir2D() {
	data_util::boolD2H(dev_dirs, h_dirs, _attrSensor2dSize);
	vector<vector<bool> > result;
	int n = 0;
	for (int i = 0; i < _attrSensorSize; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_dirs[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("dir matrix 2d get");
	return result;
}

/*
The function is getting the n power of dir matrix in 2-dimensional form
Output: npdir matrix in 2d format
*/
const vector<vector<bool> > DataManager::getNPDir2D() {
	data_util::boolD2H(dev_npdirs, h_npdirs, _npdirSize);
	vector<vector<bool> > result;
	int n = 0;
	for (int i = 0; i < _attrSensorSize; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= i + (i % 2 == 0); ++j)
			tmp.push_back(h_npdirs[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("npdir matrix 2d get");
	return result;
}

/*
The function is getting the threshold matrix in 2-dimensional form
Output: threshold matrix in 2d format
*/
const vector<vector<double> > DataManager::getThreshold2D() {
	data_util::doubleD2H(dev_thresholds, h_thresholds, _sensor2dSize);
	vector<vector<double> > result;
	int n = 0;
	for (int i = 0; i < _sensorSize; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_thresholds[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("threshold matrix 2d get");
	return result;
}

/*
The function is getting the mask amper in 2-dimension form
Output: mask amper in 2d format
*/
const vector<vector<bool> > DataManager::getMaskAmper2D() {
	vector<vector<bool> > result;
	data_util::boolD2H(dev_mask_amper, h_mask_amper, _maskAmperSize);
	int n = 0;
	for (int i = 0; i < _sensorSize; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= 2 * i + 1; ++j)
			tmp.push_back(h_mask_amper[n++]);
		result.push_back(tmp);
	}
	dataManagerLogger.debug("mask amper 2d get");
	return result;
}

/*
The function is getting mask amper as an list
Output: mask amper in 1d format
*/
const vector<bool> DataManager::getMaskAmper() {
	vector<vector<bool> > tmp = getMaskAmper2D();
	vector<bool> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	dataManagerLogger.debug("mask amper 1d get");
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
	dataManagerLogger.debug("weight matrix 1d get");
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
	dataManagerLogger.debug("dir matrix 1d get");
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
	dataManagerLogger.debug("npdir matrix 1d get");
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
	dataManagerLogger.debug("threshold matrix 1d get");
	return results;
}

/*
This function is getting the diagonal value of the weight matrix
Output: diag value
*/
const vector<double> DataManager::getDiag() {
	data_util::doubleD2H(dev_diag, h_diag, _attrSensorSize);
	vector<double> result;
	for (int i = 0; i < _attrSensorSize; ++i) {
		result.push_back(h_diag[i]);
	}
	dataManagerLogger.debug("diag value get");
	return result;
}

/*
This function is getting the diagonal value of the weight matrix of last iteration
Output: old diag value
*/
const vector<double> DataManager::getDiagOld() {
	data_util::doubleD2H(dev_diag_, h_diag_, _attrSensorSize);
	vector<double> result;
	for (int i = 0; i < _attrSensorSize; ++i) {
		result.push_back(h_diag_[i]);
	}
	dataManagerLogger.debug("old diag value get");
	return result;
}

/*
This function is getting the current mask value used in halucination
Output: mask signal
*/
const vector<bool> DataManager::getMask() {
	vector<bool> result;
	data_util::boolD2H(dev_mask, h_mask, _attrSensorSize);
	for (int i = 0; i < this->_attrSensorSize; ++i) result.push_back(h_mask[i]);
	dataManagerLogger.debug("mask signal get");
	return result;
}

/*
This function is getting the obersev matrix
Output: observe signal
*/
const vector<bool> DataManager::getObserve() {
	vector<bool> result;
	data_util::boolD2H(dev_observe, h_observe, _attrSensorSize);
	for (int i = 0; i < this->_attrSensorSize; ++i) result.push_back(h_observe[i]);
	dataManagerLogger.debug("observe signal get");
	return result;
}

/*
This function get the load signal
Output: load signal
*/
const vector<bool> DataManager::getLoad() {
	vector<bool> result;
	data_util::boolD2H(dev_load, h_load, _attrSensorSize);
	for (int i = 0; i < _attrSensorSize; ++i) result.push_back(h_load[i]);
	dataManagerLogger.debug("load signal get");
	return result;
}

const vector<bool> DataManager::getNegligible() {
	vector<bool> result;
	data_util::boolD2H(dev_negligible, h_negligible, _attrSensorSize);
	for (int i = 0; i < _attrSensorSize; ++i) {
		result.push_back(h_negligible[i]);
	}
	return result;
}

const vector<bool> DataManager::getTmpBool() {
	vector<bool> result;
	data_util::boolD2H(dev_bool_tmp, h_bool_tmp, _attrSensorSize);
	for (int i = 0; i < _attrSensorSize; ++i) {
		result.push_back(h_bool_tmp[i]);
	}
	return result;
}

/*
The function is getting the union root in union_find
Output: the union root vector
*/
const vector<int> DataManager::getUnionRoot() {
	data_util::intD2H(dev_union_root, h_union_root, _sensorSize);
	vector<int> result;
	for (int i = 0; i < _sensorSize; ++i) {
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

	s["_sensorSize"] = _sensorSize;
	s["_sensorSizeMax"] = _sensorSizeMax;
	s["_sensor2dSize"] = _sensor2dSize;
	s["_sensor2dSizeMax"] = _sensor2dSizeMax;
	s["_attrSensorSize"] = _attrSensorSize;
	s["_attrSensorSizeMax"] = _attrSensorSizeMax;
	s["_attrSensor2dSize"] = _attrSensor2dSize;
	s["_attrSensor2dSizeMax"] = _attrSensor2dSizeMax;
	s["_maskAmperSize"] = _maskAmperSize;
	s["_maskAmperSizeMax"] = _maskAmperSizeMax;
	s["_npdirSize"] = _npdirSize;
	s["_npdirSizeMax"] = _npdirSizeMax;

	return s;
}

const std::map<string, int> DataManager::convertSizeInfo(std::map<string, int> &sizeInfo) {
	std::map<string, int> s;

	s["_sensor_size"] = sizeInfo["_sensorSize"];
	s["_sensor_size_max"] = sizeInfo["_sensorSizeMax"];
	s["_sensor2d_size"] = sizeInfo["_sensor2dSize"];
	s["_sensor2d_size_max"] = sizeInfo["_sensor2dSizeMax"];
	s["_attr_sensor_size"] = sizeInfo["_attrSensorSize"];
	s["_attr_sensor_size_max"] = sizeInfo["_attrSensorSizeMax"];
	s["_attr_sensor2d_size"] = sizeInfo["_attrSensor2dSize"];
	s["_attr_sensor2d_size_max"] = sizeInfo["_attrSensor2dSizeMax"];
	s["_mask_amper_size"] = sizeInfo["_maskAmperSize"];
	s["_mask_amper_size_max"] = sizeInfo["_maskAmperSizeMax"];
	s["_npdir_size"] = sizeInfo["_npdirSize"];
	s["_npdir_size_max"] = sizeInfo["_npdirSizeMax"];

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
	case SUM: return dev_sum;
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
	case BOOL_TMP: return dev_bool_tmp;
	case CURRENT: return dev_current;
	case OLD_CURRENT: return dev_current_;
	case MASK: return dev_mask;
	case MASK_AMPER: return dev_mask_amper;
	case NPDIR_MASK: return dev_npdir_mask;
	case TARGET: return dev_target;
	case PREDICTION: return dev_prediction;
	case NEGLIGIBLE: return dev_negligible;
	case DEC_TMP1: return dev_dec_tmp1;
	case DEC_TMP2: return dev_dec_tmp2;
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
	case BOOL_TMP: return h_bool_tmp;
	case CURRENT: return h_current;
	case OLD_CURRENT: return h_current_;
	case MASK: return h_mask;
	case MASK_AMPER: return h_mask_amper;
	case PREDICTION: return h_prediction;
	case NPDIR_MASK: return h_npdir_mask;
	case TARGET: return h_target;
	case NEGLIGIBLE: return h_negligible;
	}
}