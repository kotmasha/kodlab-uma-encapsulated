#include "Simulation.h"
#include "UMACoreConstant.h"
#include "UMAutil.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "AttrSensorPair.h"
#include "Logger.h"
#include "data_util.h"
#include "kernel_util.cuh"
#include "uma_base.cuh"

extern int ind(int row, int col);
static Logger simulationLogger("Simulation", "log/simulation.log");

void simulation::abductionOverNegligible(Snapshot *snapshot, vector<vector<bool>> &sensors_to_be_added, vector<bool> &sensors_to_be_removed) {
	DataManager *dm = snapshot->getDM();
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info.at("_sensorSize");
	int attr_sensor_size = size_info.at("_attrSensorSize");
	int initial_size = snapshot->getInitialSize();

	kernel_util::initMaskSignal(dm->_dvar_b(DataManager::BOOL_TMP), initial_size * 2, attr_sensor_size);
	kernel_util::subtraction(dm->_dvar_b(DataManager::NEGLIGIBLE), dm->_dvar_b(DataManager::BOOL_TMP), attr_sensor_size);
	
	vector<bool> negligible = dm->getNegligible();
	vector<int> lists;
	for (int i = 0; i < negligible.size(); i += 2) {
		if (negligible[i]) lists.push_back(i / 2);
	}

	vector<vector<int> > dist;
	for (int i = 0; i < lists.size(); ++i) {
		vector<int> tmp;
		for (int j = 0; j <= i; ++j) {
			data_util::boolD2D(dm->_dvar_b(DataManager::NPDIR_MASK), dm->_dvar_b(DataManager::DEC_TMP1), sensor_size, i * sensor_size);
			data_util::boolD2D(dm->_dvar_b(DataManager::NPDIR_MASK), dm->_dvar_b(DataManager::DEC_TMP2), sensor_size, j * sensor_size);
			tmp.push_back(simulation::distance(dm, dm->_dvar_b(DataManager::DEC_TMP1), dm->_dvar_b(DataManager::DEC_TMP2), sensor_size));
		}
		dist.push_back(tmp);
	}
	for (int i = 0; i < lists.size(); ++i) {
		for (int j = i + 1; j < lists.size(); ++j) {
			int v = dist[j][i];
			dist[i].push_back(v);
		}
	}
	
	dm->setDists(dist);
	vector<vector<int> > groups = simulation::blocksGPU(dm, 1);
	vector<vector<bool>> inputs;

	for (int i = 0; i < groups.size(); ++i) {
		vector<AttrSensor*> m;
		for (int j = 0; j < groups[i].size(); ++j) m.push_back(snapshot->getAttrSensor(2 * groups[i][j]));
		inputs.push_back(snapshot->generateSignal(m));
	}
	vector<vector<vector<bool>>> results = simulation::abduction(dm, inputs);

	for (int i = 0; i < results.size(); ++i) {
		for (int j = 0; j < results[i].size(); ++j)
			sensors_to_be_added.push_back(results[i][j]);
	}

	sensors_to_be_removed = negligible;
}

void simulation::abductionOverDelayedSensors(Snapshot *snapshot, vector<vector<bool>> &sensors_to_be_added, vector<bool> &sensors_to_be_removed) {
	DataManager *dm = snapshot->getDM();
	int initial_size = snapshot->getInitialSize();
	int attr_sensor_size = dm->getSizeInfo().at("_attrSensorSize");

	vector<vector<bool>> downs;
	for (int i = 0; i < initial_size * 2; ++i) {
		vector<AttrSensor*> m(1, snapshot->getAttrSensor(i));
		downs.push_back(snapshot->generateSignal(m));
	}
	dm->setSignals(downs);
	simulation::downsGPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, initial_size * 2, attr_sensor_size);

	kernel_util::initMaskSignal(dm->_dvar_b(DataManager::BOOL_TMP), initial_size * 2, attr_sensor_size);
	for (int i = 0; i < initial_size * 2; ++i) {
		kernel_util::subtraction(dm->_dvar_b(DataManager::SIGNALS) + i * attr_sensor_size, dm->_dvar_b(DataManager::BOOL_TMP), attr_sensor_size);
	}
	vector<vector<bool>> delayed_downs = dm->getSignals(initial_size * 2);

	vector<vector<vector<bool>>> results = simulation::abduction(dm, delayed_downs);
	for (int i = 0; i < results.size(); ++i) {
		for (int j = 0; j < results[i].size(); ++j)
			sensors_to_be_added.push_back(results[i][j]);
	}

	for (int i = 0; i < sensors_to_be_removed.size(); ++i) dm->_hvar_b(DataManager::BOOL_TMP)[i] = sensors_to_be_removed[i];
	data_util::boolH2D(dm->_hvar_b(DataManager::BOOL_TMP), dm->_dvar_b(DataManager::BOOL_TMP), attr_sensor_size);
	for (int i = 0; i < delayed_downs.size(); ++i) {
		for(int j = 0; j < delayed_downs[i].size(); ++j) dm->_hvar_b(DataManager::SIGNALS)[i * attr_sensor_size + j] = delayed_downs[i][j];
		data_util::boolH2D(dm->_hvar_b(DataManager::SIGNALS) + i * attr_sensor_size, dm->_dvar_b(DataManager::SIGNALS) + i * attr_sensor_size, attr_sensor_size);
		kernel_util::disjunction(dm->_dvar_b(DataManager::BOOL_TMP), dm->_dvar_b(DataManager::SIGNALS) + i * attr_sensor_size, attr_sensor_size);
	}
	sensors_to_be_removed = dm->getTmpBool();
}

void simulation::enrichment(Snapshot *snapshot, bool do_pruning) {
	DataManager *dm = snapshot->getDM();
	int initial_size = snapshot->getInitialSize();
	int attr_sensor_size = dm->getSizeInfo().at("_attrSensorSize");
	int sensor_size = dm->getSizeInfo().at("_sensorSize");
	vector<vector<bool>> sensors_to_be_added;
	vector<bool> sensors_to_be_removed;
	//vector<bool> sensors_not_to_be_removed;

	data_util::boolD2D(dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::BOOL_TMP), attr_sensor_size);
	kernel_util::subtraction(dm->_dvar_b(DataManager::BOOL_TMP), dm->_dvar_b(DataManager::PREDICTION), attr_sensor_size);
	kernel_util::bool2double(dm->_dvar_b(DataManager::BOOL_TMP), dm->_dvar_d(DataManager::SUM), attr_sensor_size);
	int sum = (int)kernel_util::sum(dm->_dvar_d(DataManager::SUM), attr_sensor_size);
	if (sum > 0) {
		kernel_util::initMaskSignal(dm->_dvar_b(DataManager::BOOL_TMP), initial_size, attr_sensor_size);
		kernel_util::conjunction(dm->_dvar_b(DataManager::OLD_CURRENT), dm->_dvar_b(DataManager::BOOL_TMP), attr_sensor_size);
		sensors_to_be_added = { dm->getOldCurrent() };
	}

	if (!do_pruning) {
		vector<std::pair<string, string>> p;
		snapshot->delays(sensors_to_be_added, p);
		return;
	}

	simulation::propagateMask(dm);
	simulation::abductionOverNegligible(snapshot, sensors_to_be_added, sensors_to_be_removed);
	simulation::abductionOverDelayedSensors(snapshot, sensors_to_be_added, sensors_to_be_removed);

	std::set<size_t> sensors_to_be_added_hash;
	hash<vector<bool>> h;
	for (int i = 0; i < sensors_to_be_added.size(); ++i) {
		sensors_to_be_added_hash.insert(h(sensors_to_be_added[i]));
	}

	vector<int> sensors_to_be_removed_idx = SignalUtil::boolSignalToIntIdx(sensors_to_be_removed);
	for (int i = 0; i < sensors_to_be_removed_idx.size(); ++i) {
		vector<bool> tmp = SignalUtil::intIdxToBoolSignal(vector<int>(1, sensors_to_be_removed_idx[i]), attr_sensor_size);
		size_t tmp_hash = h(tmp);
		if (sensors_to_be_added_hash.find(tmp_hash) != sensors_to_be_added_hash.end()) {
			sensors_to_be_removed[sensors_to_be_removed_idx[i]] = false;
		}
	}

	vector<std::pair<string, string>> p;
	snapshot->delays(sensors_to_be_added, p);
	snapshot->pruning(sensors_to_be_removed);
}

float simulation::decide(Snapshot *snapshot, vector<bool> &signal, const double &phi, const bool active) {//the decide function
	DataManager *dm = snapshot->getDM();
	double total = snapshot->getTotal();
	double total_ = snapshot->getOldTotal();
	double q = snapshot->getQ();
	int initial_size = snapshot->getInitialSize();
	int attr_sensor_size = dm->getSizeInfo().at("_attrSensorSize");

	data_util::boolD2D(dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::OLD_CURRENT), attr_sensor_size);

	snapshot->generateObserve(signal);

	snapshot->updateTotal(phi, active);
	
	if (UMA_SNAPSHOT::SNAPSHOT_STATIONARY == snapshot->getType())
		simulation::updateState(dm, q, phi, total, total_, active);
	else if (UMA_SNAPSHOT::SNAPSHOT_QUALITATIVE == snapshot->getType())
		simulation::updateStateQualitative(dm, q, phi, total, total_, active);
	else if (UMA_SNAPSHOT::SNAPSHOT_DISCOUNTED == snapshot->getType())
		simulation::updateStateDiscounted(dm, q, phi, total, total_, active);
	else if (UMA_SNAPSHOT::SNAPSHOT_EMPIRICAL == snapshot->getType())
		simulation::updateStateEmpirical(dm, q, phi, total, total_, active);
	else {//type empirical
		throw UMAInternalException("Requested snapshot type is undefined!\n", true, &simulationLogger);
	}
	snapshot->updateQ();
	simulation::halucinate(dm, initial_size);

	if (snapshot->getAutoTarget()) simulation::calculateTarget(dm, snapshot->getType());

	return simulation::divergence(dm);
}

vector<float> simulation::decide(Agent *agent, vector<bool> &obs_plus, vector<bool> &obs_minus, const double &phi, const bool &active) {//the decide function
	vector<float> results;
	Snapshot *snapshot_plus = agent->getSnapshot("plus");
	Snapshot *snapshot_minus = agent->getSnapshot("minus");

	if (agent->getEnableEnrichment()) {
		bool doPruning = agent->doPruning();
		if (active) simulation::enrichment(snapshot_plus, doPruning);
		else simulation::enrichment(snapshot_minus, doPruning);
	}

	const float res_plus = simulation::decide(snapshot_plus, obs_plus, phi, active);
	const float res_minus = simulation::decide(snapshot_minus, obs_minus, phi, !active);
	
	results.push_back(res_plus);
	results.push_back(res_minus);

	agent->setT(agent->getT() + 1);
	return results;
}

void simulation::updateState(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];

	//update weights
	uma_base::updateWeights(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_b(DataManager::OBSERVE), attr_sensor_size, q, phi, active);
	//update diag from new weights
	uma_base::getWeightsDiag(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::DIAG), dm->_dvar_d(DataManager::OLD_DIAG), attr_sensor_size);
	//update thresholds
	uma_base::updateThresholds(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::THRESHOLDS), total_, q, phi, sensor_size);
	//update dirs
	uma_base::orientAll(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::THRESHOLDS), total, sensor_size);
	//floyd on new dirs and get npdirs
	floyd(dm);
	//negligible on npdir, store the results
	uma_base::negligible(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::NEGLIGIBLE), sensor_size);

	//the propagation on 
	data_util::boolD2D(dm->_dvar_b(DataManager::OBSERVE), dm->_dvar_b(DataManager::SIGNALS), attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::CURRENT), 1, attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::CURRENT), dm->_hvar_b(DataManager::CURRENT), attr_sensor_size);
}

void simulation::updateStateQualitative(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];

	//update weights
	uma_base_qualitative::updateWeights(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_b(DataManager::OBSERVE), attr_sensor_size, q, phi, active);
	//update diag from new weights
	uma_base::getWeightsDiag(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::DIAG), dm->_dvar_d(DataManager::OLD_DIAG), attr_sensor_size);
	//update thresholds, no operation for now
	//uma_base::update_thresholds(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::THRESHOLDS), total_, q, phi, sensor_size);
	//update dirs
	uma_base_qualitative::orientAll(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::THRESHOLDS), total, sensor_size);
	//floyd on new dirs and get npdirs
	floyd(dm);
	//negligible on npdir, store the results
	uma_base::negligible(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::NEGLIGIBLE), sensor_size);

	//the propagation on 
	data_util::boolD2D(dm->_dvar_b(DataManager::OBSERVE), dm->_dvar_b(DataManager::SIGNALS), attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::CURRENT), 1, attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::CURRENT), dm->_hvar_b(DataManager::CURRENT), attr_sensor_size);
}

void simulation::updateStateDiscounted(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];

	//update weights
	uma_base_discounted::updateWeights(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_b(DataManager::OBSERVE), attr_sensor_size, q, phi, active);
	//update diag from new weights
	uma_base::getWeightsDiag(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::DIAG), dm->_dvar_d(DataManager::OLD_DIAG), attr_sensor_size);
	//update thresholds, no operation for now
	//uma_base::update_thresholds(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::THRESHOLDS), total_, q, phi, sensor_size);
	//update dirs
	uma_base_discounted::orientAll(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::THRESHOLDS), total, sensor_size);
	//floyd on new dirs and get npdirs
	floyd(dm);
	//negligible on npdir, store the results
	uma_base::negligible(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::NEGLIGIBLE), sensor_size);

	//the propagation on 
	data_util::boolD2D(dm->_dvar_b(DataManager::OBSERVE), dm->_dvar_b(DataManager::SIGNALS), attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::CURRENT), 1, attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::CURRENT), dm->_hvar_b(DataManager::CURRENT), attr_sensor_size);
}

void simulation::updateStateEmpirical(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];

	//update weights
	uma_base_empirical::updateWeights(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_b(DataManager::OBSERVE), attr_sensor_size, q, phi, active);
	//update diag from new weights
	uma_base::getWeightsDiag(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::DIAG), dm->_dvar_d(DataManager::OLD_DIAG), attr_sensor_size);
	//update thresholds, no operation for now
	//uma_base::update_thresholds(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::THRESHOLDS), total_, q, phi, sensor_size);
	//update dirs
	uma_base_empirical::orientAll(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::THRESHOLDS), total, sensor_size);
	//floyd on new dirs and get npdirs
	floyd(dm);
	//negligible on npdir, store the results
	uma_base::negligible(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::NEGLIGIBLE), sensor_size);

	//the propagation on 
	data_util::boolD2D(dm->_dvar_b(DataManager::OBSERVE), dm->_dvar_b(DataManager::SIGNALS), attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::CURRENT), 1, attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::CURRENT), dm->_hvar_b(DataManager::CURRENT), attr_sensor_size);
}

void simulation::floyd(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];
	int attr_sensor2d_size = size_info["_attrSensor2dSize"];
	uma_base::copyNpdir(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::DIRS), attr_sensor_size);
	uma_base::floyd(dm->_dvar_b(DataManager::NPDIRS), attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::NPDIRS), dm->_hvar_b(DataManager::NPDIRS), attr_sensor2d_size);
	//cudaCheckErrors("kernel fails");
	simulationLogger.debug("floyd is done");
}

void simulation::propagates(bool *npdirs, bool *load, bool *signals, bool *lsignals, bool *dst, int sig_count, int attr_sensor_size) {
	//prepare loaded signals
	data_util::boolD2D(signals, lsignals, sig_count * attr_sensor_size);

	for (int i = 0; i < sig_count; ++i) {
		kernel_util::disjunction(lsignals + i * attr_sensor_size, load, attr_sensor_size);
	}
	//propagate using npdir
	uma_base::transposeMultiply(npdirs, lsignals, attr_sensor_size, sig_count);
	uma_base::transposeMultiply(npdirs, signals, attr_sensor_size, sig_count);
	kernel_util::negateConjunctionStar(lsignals, signals, sig_count * attr_sensor_size);
	//copy result to dst
	if (dst != NULL) {
		data_util::boolD2D(lsignals, dst, sig_count * attr_sensor_size);
	}
	simulationLogger.debug("Propagation is done for " + to_string(sig_count) + " sensors");
}

void simulation::halucinate(DataManager *dm, int &initial_size) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];
	simulation::genMask(dm, initial_size);

	data_util::boolD2D(dm->_dvar_b(DataManager::MASK), dm->_dvar_b(DataManager::SIGNALS), attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::PREDICTION), 1, attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::PREDICTION), dm->_hvar_b(DataManager::PREDICTION), attr_sensor_size);

	simulationLogger.debug("Halucniate is done");
}

void simulation::genMask(DataManager *dm,  int &initial_size) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];
	uma_base::initMask(dm->_dvar_b(DataManager::MASK), initial_size, attr_sensor_size);

	uma_base::mask(dm->_dvar_b(DataManager::MASK_AMPER), dm->_dvar_b(DataManager::MASK), dm->_dvar_b(DataManager::CURRENT), sensor_size);
	uma_base::checkMask(dm->_dvar_b(DataManager::MASK), sensor_size);
	simulationLogger.debug("Mask is generated");
}

void simulation::propagateMask(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::NPDIR_MASK), sensor_size * attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::SIGNALS), sensor_size * attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
	for (int i = 0; i < sensor_size; ++i) {
		data_util::boolD2D(dm->_dvar_b(DataManager::MASK_AMPER), dm->_dvar_b(DataManager::SIGNALS), (ind(i + 1, 0) - ind(i, 0)) * 2, ind(i, 0) * 2, attr_sensor_size * i);
	}
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::NPDIR_MASK), NULL, sensor_size, attr_sensor_size);
}

void simulation::calculateTarget(DataManager *dm, const int type) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];

	if (UMA_AGENT::AGENT_STATIONARY == type)
		uma_base::calculateTarget(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::TARGET), sensor_size);
	else if(UMA_AGENT::AGENT_QUALITATIVE == type){
		data_util::doubleD2H(dm->_dvar_d(DataManager::DIAG), dm->_hvar_d(DataManager::DIAG), attr_sensor_size);
		uma_base_qualitative::calculateTarget(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::TARGET), sensor_size);
	}
	else if (UMA_AGENT::AGENT_DISCOUNTED == type) {
		data_util::doubleD2H(dm->_dvar_d(DataManager::DIAG), dm->_hvar_d(DataManager::DIAG), attr_sensor_size);
		uma_base_discounted::calculateTarget(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::TARGET), sensor_size);
	}
	else {
		data_util::doubleD2H(dm->_dvar_d(DataManager::DIAG), dm->_hvar_d(DataManager::DIAG), attr_sensor_size);
		uma_base_empirical::calculateTarget(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::TARGET), sensor_size);
	}
	data_util::boolD2H(dm->_dvar_b(DataManager::TARGET), dm->_hvar_b(DataManager::TARGET), attr_sensor_size);
}

/*
This function get the diff of current and target
*/
float simulation::divergence(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attrSensorSize"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);

	data_util::boolD2H(dm->_dvar_b(DataManager::PREDICTION), dm->_hvar_b(DataManager::PREDICTION), attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::TARGET), dm->_hvar_b(DataManager::TARGET), attr_sensor_size);

	data_util::boolD2D(dm->_dvar_b(DataManager::PREDICTION), dm->_dvar_b(DataManager::DEC_TMP1), attr_sensor_size);
	data_util::boolD2D(dm->_dvar_b(DataManager::TARGET), dm->_dvar_b(DataManager::DEC_TMP2), attr_sensor_size);

	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::DEC_TMP1), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::DEC_TMP1), 1, attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::DEC_TMP2), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::DEC_TMP2), 1, attr_sensor_size);

	kernel_util::subtraction(dm->_dvar_b(DataManager::DEC_TMP1), dm->_dvar_b(DataManager::DEC_TMP2), attr_sensor_size);
	//apply weights to the computed divergence signal and output the result:

	//uma_base::delta_weight_sum(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::DEC_TMP1), dm->_dvar_f(DataManager::RES), attr_sensor_size);
	//data_util::floatD2H(dm->_dvar_f(DataManager::RES), dm->_hvar_f(DataManager::RES), 1);

	//const float value = *(dm->_hvar_f(DataManager::RES));

	kernel_util::bool2double(dm->_dvar_b(DataManager::DEC_TMP1), dm->_dvar_d(DataManager::SUM), attr_sensor_size);
	int value = kernel_util::sum(dm->_dvar_d(DataManager::SUM), attr_sensor_size);

	return value;
}

//calculate the distance between 2 bool array
int simulation::distance(DataManager *dm, bool *b1, bool *b2, int size) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attrSensorSize"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);

	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), b1, dm->_dvar_b(DataManager::LSIGNALS), NULL, 1, attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), b2, dm->_dvar_b(DataManager::LSIGNALS), NULL, 1, attr_sensor_size);

	kernel_util::ConjunctionStar(b1, b2, attr_sensor_size);

	kernel_util::bool2double(b1, dm->_dvar_d(DataManager::SUM), attr_sensor_size);
	int sum = (int)kernel_util::sum(dm->_dvar_d(DataManager::SUM), attr_sensor_size);

	return sum;
}

void simulation::upsGPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int attr_sensor_size) {
	uma_base::transposeMultiply(npdirs, signals, attr_sensor_size, sig_count);
	if (dst != NULL) {
		data_util::boolD2D(signals, dst, sig_count * attr_sensor_size);
	}
}

void simulation::downsGPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int attr_sensor_size) {
	uma_base::multiply(npdirs, signals, attr_sensor_size, sig_count);
	if (dst != NULL) {
		data_util::boolD2D(signals, dst, sig_count * attr_sensor_size);
	}
}

vector<vector<vector<bool> > > simulation::abduction(DataManager *dm, const vector<vector<bool> > &signals) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attrSensorSize"];

	vector<vector<vector<bool> > > results;
	vector<vector<bool> > even_results, odd_results;
	for (int i = 0; i < signals.size(); ++i) {
		vector<int> even_idx, odd_idx;
		for (int j = 0; j < signals[0].size() / 2; ++j) {
			if (signals[i][2 * j]) even_idx.push_back(j);
			if (signals[i][2 * j + 1]) odd_idx.push_back(j);
		}
		if (even_idx.empty()) {
			vector<bool> tmp(attr_sensor_size, false);
			even_results.push_back(tmp);
		}
		else {
			kernel_util::alltrue(dm->_dvar_b(DataManager::SIGNALS), attr_sensor_size);
			for (int j = 0; j < even_idx.size(); ++j) {
				kernel_util::conjunction(dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::NPDIR_MASK) + even_idx[j] * attr_sensor_size, attr_sensor_size);
			}
			vector<vector<bool> > signals = dm->getSignals(1);
			even_results.push_back(signals[0]);
		}
		if (odd_idx.empty()) {
			vector<bool> tmp(attr_sensor_size, false);
			odd_results.push_back(tmp);
		}
		else {
			kernel_util::allfalse(dm->_dvar_b(DataManager::SIGNALS), attr_sensor_size);
			for (int j = 0; j < odd_idx.size(); ++j) {
				kernel_util::disjunction(dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::NPDIR_MASK) + odd_idx[j] * attr_sensor_size, attr_sensor_size);
			}

			kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
			int tmp = size_info["_npdirSize"];
			kernel_util::allfalse(dm->_dvar_b(DataManager::NPDIRS), tmp);
			simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), NULL, 1, attr_sensor_size);
			vector<vector<bool> > signals = dm->getLSignals(1);
			odd_results.push_back(signals[0]);
		}
	}
	results.push_back(even_results);
	results.push_back(odd_results);
	return results;
}

vector<vector<int> > simulation::blocksGPU(DataManager *dm, float delta) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensorSize"];
	int attr_sensor_size = size_info["_attrSensorSize"];

	int t = floor(log(sensor_size) / log(2)) + 1;
	for (int i = 0; i < t; ++i) {
		uma_base::dioidSquare(dm->_dvar_i(DataManager::DISTS), sensor_size);
	}

	uma_base::unionInit(dm->_dvar_i(DataManager::UNION_ROOT), sensor_size);
	uma_base::checkDist(dm->_dvar_i(DataManager::DISTS), delta, sensor_size);

	uma_base::unionGPU(dm->_dvar_i(DataManager::DISTS), dm->_dvar_i(DataManager::UNION_ROOT), sensor_size);
	data_util::intD2H(dm->_dvar_i(DataManager::UNION_ROOT), dm->_hvar_i(DataManager::UNION_ROOT), sensor_size);

	map<int, int> m;
	vector<vector<int> > result;
	vector<int> union_root = dm->getUnionRoot();
	for (int i = 0; i < sensor_size; ++i) {
		if (m.find(union_root[i]) == m.end()) {
			m[union_root[i]] = result.size();
			vector<int> tmp;
			tmp.push_back(i);
			result.push_back(tmp);
		}
		else {
			result[m[union_root[i]]].push_back(i);
		}
	}

	return result;
}