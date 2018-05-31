#include "Simulation.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "AttrSensorPair.h"
#include "Logger.h"
#include "data_util.h"
#include "kernel_util.cuh"
#include "uma_base.cuh"
#include "UMAutil.h"

extern int ind(int row, int col);
static Logger simulationLogger("Simulation", "log/simulation.log");

void simulation::abduction_over_negligible(Snapshot *snapshot, vector<vector<bool>> &sensors_to_be_added, vector<bool> &sensors_to_be_removed) {
	DataManager *dm = snapshot->getDM();
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info.at("_sensor_size");
	int attr_sensor_size = size_info.at("_attr_sensor_size");
	int initial_size = snapshot->getInitialSize();
	cout << "attr_sensor_size in negligible:" << attr_sensor_size << endl;

	kernel_util::init_mask_signal(dm->_dvar_b(DataManager::BOOL_TMP), initial_size * 2, attr_sensor_size);
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
	vector<vector<int> > groups = simulation::blocks_GPU(dm, 1);
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

void simulation::abduction_over_delayed_sensors(Snapshot *snapshot, vector<vector<bool>> &sensors_to_be_added, vector<bool> &sensors_to_be_removed) {
	DataManager *dm = snapshot->getDM();
	int initial_size = snapshot->getInitialSize();
	int attr_sensor_size = dm->getSizeInfo().at("_attr_sensor_size");
	cout << "attr_sensor_size in delayed sensor:" << attr_sensor_size << endl;
	vector<vector<bool>> downs;
	for (int i = 0; i < initial_size * 2; ++i) {
		vector<AttrSensor*> m(1, snapshot->getAttrSensor(i));
		downs.push_back(snapshot->generateSignal(m));
	}
	dm->setSignals(downs);
	simulation::downs_GPU(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::SIGNALS), NULL, initial_size * 2, attr_sensor_size);

	kernel_util::init_mask_signal(dm->_dvar_b(DataManager::BOOL_TMP), initial_size * 2, attr_sensor_size);
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
	int attr_sensor_size = dm->getSizeInfo().at("_attr_sensor_size");
	int sensor_size = dm->getSizeInfo().at("_sensor_size");
	vector<vector<bool>> sensors_to_be_added;
	vector<bool> sensors_to_be_removed;
	//vector<bool> sensors_not_to_be_removed;

	data_util::boolD2D(dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::BOOL_TMP), attr_sensor_size);
	kernel_util::subtraction(dm->_dvar_b(DataManager::BOOL_TMP), dm->_dvar_b(DataManager::PREDICTION), attr_sensor_size);
	kernel_util::bool2double(dm->_dvar_b(DataManager::BOOL_TMP), dm->_dvar_d(DataManager::SUM), attr_sensor_size);
	int sum = (int)kernel_util::sum(dm->_dvar_d(DataManager::SUM), attr_sensor_size);
	if (sum > 0) {
		kernel_util::init_mask_signal(dm->_dvar_b(DataManager::BOOL_TMP), initial_size, attr_sensor_size);
		kernel_util::conjunction(dm->_dvar_b(DataManager::OLD_CURRENT), dm->_dvar_b(DataManager::BOOL_TMP), attr_sensor_size);
		sensors_to_be_added = { dm->getOldCurrent() };
	}
	//moving this line to other place?
	data_util::boolD2D(dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::OLD_CURRENT), attr_sensor_size);

	if (!do_pruning) {
		vector<std::pair<string, string>> p;
		snapshot->delays(sensors_to_be_added, p);
		return;
	}
	
	simulation::propagate_mask(dm);
	simulation::abduction_over_negligible(snapshot, sensors_to_be_added, sensors_to_be_removed);
	simulation::abduction_over_delayed_sensors(snapshot, sensors_to_be_added, sensors_to_be_removed);

	std::set<size_t> sensors_to_be_added_hash;
	hash<vector<bool>> h;
	for (int i = 0; i < sensors_to_be_added.size(); ++i) {
		sensors_to_be_added_hash.insert(h(sensors_to_be_added[i]));
	}

	vector<int> sensors_to_be_removed_idx = SignalUtil::bool_signal_to_int_idx(sensors_to_be_removed);
	for (int i = 0; i < sensors_to_be_removed_idx.size(); ++i) {
		vector<bool> tmp = SignalUtil::int_idx_to_bool_signal(vector<int>(1, sensors_to_be_removed_idx[i]), attr_sensor_size);
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

	snapshot->generateObserve(signal);

	snapshot->update_total(phi, active);
	
	if (AGENT_TYPE::STATIONARY == snapshot->getType())
		simulation::update_state(dm, q, phi, total, total_, active);
	else
		simulation::update_state_qualitative(dm, q, phi, total, total_, active);
	simulation::halucinate(dm, initial_size);

	if (snapshot->getAutoTarget()) simulation::calculate_target(dm, snapshot->getType());

	return simulation::divergence(dm);
}

vector<float> simulation::decide(Agent *agent, vector<bool> &obs_plus, vector<bool> &obs_minus, const double &phi, const bool &active) {//the decide function
	vector<float> results;
	Snapshot *snapshot_plus = agent->getSnapshot("plus");
	Snapshot *snapshot_minus = agent->getSnapshot("minus");

	//bool do_pruning = agent->do_pruning();
	//if (active) simulation::enrichment(snapshot_plus, do_pruning);
	//else simulation::enrichment(snapshot_minus, do_pruning);

	const float res_plus = simulation::decide(snapshot_plus, obs_plus, phi, active);
	const float res_minus = simulation::decide(snapshot_minus, obs_minus, phi, !active);

	cout << "divergence result: " <<res_plus << "," << res_minus << endl;
	
	results.push_back(res_plus);
	results.push_back(res_minus);

	agent->setT(agent->getT() + 1);
	return results;
}

void simulation::update_state(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int attr_sensor_size = size_info["_attr_sensor_size"];

	//update weights
	uma_base::update_weights(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_b(DataManager::OBSERVE), attr_sensor_size, q, phi, active);
	//update diag from new weights
	uma_base::get_weights_diag(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::DIAG), dm->_dvar_d(DataManager::OLD_DIAG), attr_sensor_size);
	//update thresholds
	uma_base::update_thresholds(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::THRESHOLDS), total_, q, phi, sensor_size);
	//update dirs
	uma_base::orient_all(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::WEIGHTS), total, sensor_size);
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

void simulation::update_state_qualitative(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int attr_sensor_size = size_info["_attr_sensor_size"];

	//update weights
	uma_base_qualitative::update_weights(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_b(DataManager::OBSERVE), attr_sensor_size, q, phi, active);
	//update diag from new weights
	uma_base::get_weights_diag(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::DIAG), dm->_dvar_d(DataManager::OLD_DIAG), attr_sensor_size);
	//update thresholds, no operation for now
	//uma_base::update_thresholds(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::THRESHOLDS), total_, q, phi, sensor_size);
	//update dirs
	uma_base_qualitative::orient_all(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::WEIGHTS), total, sensor_size);
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
	int sensor_size = size_info["_sensor_size"];
	int attr_sensor_size = size_info["_attr_sensor_size"];
	int attr_sensor2d_size = size_info["_attr_sensor2d_size"];
	uma_base::copy_npdir(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::DIRS), attr_sensor_size);
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
	uma_base::transpose_multiply(npdirs, lsignals, attr_sensor_size, sig_count);
	uma_base::transpose_multiply(npdirs, signals, attr_sensor_size, sig_count);
	kernel_util::negate_conjunction_star(lsignals, signals, sig_count * attr_sensor_size);
	//copy result to dst
	if (dst != NULL) {
		data_util::boolD2D(lsignals, dst, sig_count * attr_sensor_size);
	}
	simulationLogger.debug("Propagation is done for " + to_string(sig_count) + " sensors");
}

void simulation::halucinate(DataManager *dm, int &initial_size) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int attr_sensor_size = size_info["_attr_sensor_size"];
	simulation::gen_mask(dm, initial_size);

	data_util::boolD2D(dm->_dvar_b(DataManager::MASK), dm->_dvar_b(DataManager::SIGNALS), attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::PREDICTION), 1, attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::PREDICTION), dm->_hvar_b(DataManager::PREDICTION), attr_sensor_size);

	//data_util::boolD2H(dm->_dvar_b(DataManager::MASK), dm->_hvar_b(DataManager::MASK), attr_sensor_size);
	//for (int i = 0; i < attr_sensor_size; ++i) cout << dm->_hvar_b(DataManager::MASK)[i] << ",";
	//cout << endl;
	simulationLogger.debug("Halucniate is done");
}

void simulation::gen_mask(DataManager *dm,  int &initial_size) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int attr_sensor_size = size_info["_attr_sensor_size"];
	uma_base::init_mask(dm->_dvar_b(DataManager::MASK), initial_size, attr_sensor_size);

	uma_base::mask(dm->_dvar_b(DataManager::MASK_AMPER), dm->_dvar_b(DataManager::MASK), dm->_dvar_b(DataManager::CURRENT), sensor_size);
	uma_base::check_mask(dm->_dvar_b(DataManager::MASK), sensor_size);
	simulationLogger.debug("Mask is generated");
}

void simulation::propagate_mask(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int attr_sensor_size = size_info["_attr_sensor_size"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::NPDIR_MASK), sensor_size * attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::SIGNALS), sensor_size * attr_sensor_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);
	for (int i = 0; i < sensor_size; ++i) {
		data_util::boolD2D(dm->_dvar_b(DataManager::MASK_AMPER), dm->_dvar_b(DataManager::SIGNALS), (ind(i + 1, 0) - ind(i, 0)) * 2, ind(i, 0) * 2, attr_sensor_size * i);
	}
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::NPDIR_MASK), NULL, sensor_size, attr_sensor_size);
}

void simulation::calculate_target(DataManager *dm, const int type) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int attr_sensor_size = size_info["_attr_sensor_size"];

	if (AGENT_TYPE::STATIONARY == type)
		uma_base::calculate_target(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::TARGET), sensor_size);
	else {
		data_util::doubleD2H(dm->_dvar_d(DataManager::DIAG), dm->_hvar_d(DataManager::DIAG), attr_sensor_size);
		uma_base_qualitative::calculate_target(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::TARGET), sensor_size);
	}
	data_util::boolD2H(dm->_dvar_b(DataManager::TARGET), dm->_hvar_b(DataManager::TARGET), attr_sensor_size);
}

/*
This function get the diff of current and target
*/
float simulation::divergence(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attr_sensor_size"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);

	data_util::boolD2H(dm->_dvar_b(DataManager::PREDICTION), dm->_hvar_b(DataManager::PREDICTION), attr_sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::TARGET), dm->_hvar_b(DataManager::TARGET), attr_sensor_size);
	//for (int i = 0; i < attr_sensor_size; ++i) cout << dm->_hvar_b(DataManager::PREDICTION)[i] << ",";
	//cout << endl;
	//for (int i = 0; i < attr_sensor_size; ++i) cout << dm->_hvar_b(DataManager::TARGET)[i] << ",";
	//cout << endl;

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
	int attr_sensor_size = size_info["_attr_sensor_size"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), attr_sensor_size);

	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), b1, dm->_dvar_b(DataManager::LSIGNALS), NULL, 1, attr_sensor_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), b2, dm->_dvar_b(DataManager::LSIGNALS), NULL, 1, attr_sensor_size);

	kernel_util::conjunction_star(b1, b2, attr_sensor_size);

	kernel_util::bool2double(b1, dm->_dvar_d(DataManager::SUM), attr_sensor_size);
	int sum = (int)kernel_util::sum(dm->_dvar_d(DataManager::SUM), attr_sensor_size);

	return sum;
}

void simulation::ups_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int attr_sensor_size) {
	uma_base::transpose_multiply(npdirs, signals, attr_sensor_size, sig_count);
	if (dst != NULL) {
		data_util::boolD2D(signals, dst, sig_count * attr_sensor_size);
	}
}

void simulation::downs_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int attr_sensor_size) {
	uma_base::multiply(npdirs, signals, attr_sensor_size, sig_count);
	if (dst != NULL) {
		data_util::boolD2D(signals, dst, sig_count * attr_sensor_size);
	}
}

vector<vector<vector<bool> > > simulation::abduction(DataManager *dm, const vector<vector<bool> > &signals) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int attr_sensor_size = size_info["_attr_sensor_size"];
	cout << "attr_sensor_size:" << attr_sensor_size << endl;

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
			int tmp = size_info["_npdir_size"];
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

vector<vector<int> > simulation::blocks_GPU(DataManager *dm, float delta) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int attr_sensor_size = size_info["_attr_sensor_size"];

	int t = floor(log(sensor_size) / log(2)) + 1;
	for (int i = 0; i < t; ++i) {
		uma_base::dioid_square(dm->_dvar_i(DataManager::DISTS), sensor_size);
	}

	uma_base::union_init(dm->_dvar_i(DataManager::UNION_ROOT), sensor_size);
	uma_base::check_dist(dm->_dvar_i(DataManager::DISTS), delta, sensor_size);

	uma_base::union_GPU(dm->_dvar_i(DataManager::DISTS), dm->_dvar_i(DataManager::UNION_ROOT), sensor_size);
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