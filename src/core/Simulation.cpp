#include "Simulation.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "MeasurablePair.h"
#include "Logger.h"
#include "data_util.h"
#include "kernel_util.cuh"
#include "uma_base.cuh"

extern int ind(int row, int col);
static Logger simulationLogger("Simulation", "log/simulation.log");


void simulation::update_total(double &total, double &total_, const double &q, const double &phi) {
	total_ = total;
	total = q * total + (1 - q) * phi;
}

float simulation::decide(Snapshot *snapshot, const vector<bool> &signal, const double &phi, const bool active) {//the decide function
	DataManager *dm = snapshot->getDM();
	double total = snapshot->getTotal();
	double total_ = snapshot->getOldTotal();
	double q = snapshot->getQ();
	int initial_size = snapshot->getInitialSize();

	dm->setObserve(signal);
	simulation::update_total(total, total_, q, phi);
	snapshot->setTotal(total);
	snapshot->setOldTotal(total_);
	simulation::update_state(dm, q, phi, total, total_, active);
	simulation::halucinate(dm, initial_size);

	if (snapshot->getPropagateMask()) simulation::propagate_mask(dm);
	if (snapshot->getAutoTarget()) simulation::calculate_target(dm);

	return simulation::divergence(dm);
}

vector<float> simulation::decide(Agent *agent, const vector<bool> &obs_plus, const vector<bool> &obs_minus, const double &phi, const bool &active) {//the decide function
	vector<float> results;
	Snapshot *snapshot_plus = agent->getSnapshot("plus");
	Snapshot *snapshot_minus = agent->getSnapshot("minus");
	const float res_plus = simulation::decide(snapshot_plus, obs_plus, phi, active);
	const float res_minus = simulation::decide(snapshot_minus, obs_minus, phi, !active);
	
	results.push_back(res_plus);
	results.push_back(res_minus);

	return results;
}

void simulation::update_state(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int measurable_size = size_info["_measurable_size"];
	//update weights
	uma_base::update_weights(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_b(DataManager::OBSERVE), measurable_size, q, phi, active);
	//update diag from new weights
	uma_base::get_weights_diag(dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::DIAG), dm->_dvar_d(DataManager::OLD_DIAG), measurable_size);
	//update thresholds
	uma_base::update_thresholds(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::THRESHOLDS), total_, q, phi, sensor_size);
	//update dirs
	uma_base::orient_all(dm->_dvar_b(DataManager::DIRS), dm->_dvar_d(DataManager::WEIGHTS), dm->_dvar_d(DataManager::WEIGHTS), total, sensor_size);
	//floyd on new dirs and get npdirs
	floyd(dm);
	//negligible on npdir, store the results
	uma_base::negligible(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::NEGLIGIBLE), sensor_size);

	//the propagation on 
	data_util::boolD2D(dm->_dvar_b(DataManager::OBSERVE), dm->_dvar_b(DataManager::SIGNALS), measurable_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), measurable_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::CURRENT), 1, measurable_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::CURRENT), measurable_size);
}

void simulation::floyd(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int measurable_size = size_info["_measurable_size"];
	int measurable2d_size = size_info["_measurable2d_size"];
	uma_base::copy_npdir(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::DIRS), measurable_size);
	uma_base::floyd(dm->_dvar_b(DataManager::NPDIRS), measurable_size);
	data_util::boolD2H(dm->_hvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::NPDIRS), measurable2d_size);
	//cudaCheckErrors("kernel fails");
	simulationLogger.debug("floyd is done");
}

void simulation::propagates(bool *npdirs, bool *load, bool *signals, bool *lsignals, bool *dst, int sig_count, int measurable_size) {
	//prepare loaded signals
	data_util::boolD2D(signals, lsignals, sig_count * measurable_size);
	for (int i = 0; i < sig_count; ++i) {
		kernel_util::disjunction(lsignals + i * measurable_size, load, measurable_size);
	}
	//propagate using npdir
	uma_base::transpose_multiply(npdirs, lsignals, measurable_size, sig_count);
	uma_base::transpose_multiply(npdirs, signals, measurable_size, sig_count);
	kernel_util::negate_conjunction_star(lsignals, signals, sig_count * measurable_size);
	//copy result to dst
	if (dst != NULL) {
		data_util::boolD2D(lsignals, dst, sig_count * measurable_size);
	}
	simulationLogger.debug("Propagation is done for " + to_string(sig_count) + " sensors");
}

void simulation::halucinate(DataManager *dm, int &initial_size) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int measurable_size = size_info["_measurable_size"];
	simulation::gen_mask(dm, initial_size);

	data_util::boolD2D(dm->_dvar_b(DataManager::MASK), dm->_dvar_b(DataManager::SIGNALS), measurable_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), measurable_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::PREDICTION), 1, measurable_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::PREDICTION), dm->_hvar_b(DataManager::PREDICTION), measurable_size);
	simulationLogger.debug("Halucniate is done");
}

void simulation::gen_mask(DataManager *dm,  int &initial_size) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int measurable_size = size_info["_measurable_size"];
	uma_base::init_mask(dm->_dvar_b(DataManager::MASK), initial_size, measurable_size);

	uma_base::mask(dm->_dvar_b(DataManager::MASK_AMPER), dm->_dvar_b(DataManager::MASK), dm->_dvar_b(DataManager::CURRENT), sensor_size);
	uma_base::check_mask(dm->_dvar_b(DataManager::MASK), sensor_size);
	simulationLogger.debug("Mask is generated");
}

void simulation::propagate_mask(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int measurable_size = size_info["_measurable_size"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::NPDIR_MASK), sensor_size * measurable_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::SIGNALS), sensor_size * measurable_size);
	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), measurable_size);
	for (int i = 0; i < sensor_size; ++i) {
		data_util::boolD2D(dm->_dvar_b(DataManager::MASK_AMPER), dm->_dvar_b(DataManager::SIGNALS), (ind(i + 1, 0) - ind(i, 0)) * 2, ind(i, 0) * 2, measurable_size * i);
	}
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::NPDIR_MASK), NULL, sensor_size, measurable_size);
}

void simulation::calculate_target(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int sensor_size = size_info["_sensor_size"];
	int measurable_size = size_info["_measurable_size"];

	uma_base::calculate_target(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::TARGET), sensor_size);
	data_util::boolD2H(dm->_dvar_b(DataManager::TARGET), dm->_hvar_b(DataManager::TARGET), measurable_size);
}

/*
This function get the diff of current and target
*/
float simulation::divergence(DataManager *dm) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int measurable_size = size_info["_measurable_size"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), measurable_size);

	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::CURRENT), 1, measurable_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::TARGET), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::TARGET), 1, measurable_size);

	kernel_util::subtraction(dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::TARGET), measurable_size);
	//apply weights to the computed divergence signal and output the result:

	uma_base::delta_weight_sum(dm->_dvar_d(DataManager::DIAG), dm->_dvar_b(DataManager::CURRENT), dm->_dvar_f(DataManager::RES), measurable_size);
	data_util::floatD2H(dm->_dvar_f(DataManager::RES), dm->_hvar_f(DataManager::RES), 1);

	const float value = *(dm->_hvar_f(DataManager::RES));

	return value;
}

//still have problem in the function
//deprecated
float simulation::distance(DataManager *dm) {
	/*
	std::map<string, int> size_info = dm->getSizeInfo();
	int measurable_size = size_info["_measurable_size"];

	kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), measurable_size);

	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::CURRENT), 1, measurable_size);
	simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::TARGET), dm->_dvar_b(DataManager::LSIGNALS), dm->_dvar_b(DataManager::TARGET), 1, measurable_size);

	kernel_util::conjunction_star(dm->_dvar_b(DataManager::CURRENT), dm->_dvar_b(DataManager::TARGET), measurable_size);

	//cudaMemcpy(h_signal, signal1, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int sum = 0;
	//for (int i = 0; i < _measurable_size; ++i) sum += h_signal[i];
	return sum;
	*/
	return 0.0;
}

void simulation::ups_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int measurable_size) {
	uma_base::transpose_multiply(npdirs, signals, measurable_size, sig_count);
	if (dst != NULL) {
		data_util::boolD2D(signals, dst, sig_count * measurable_size);
	}
}

void simulation::downs_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int measurable_size) {
	uma_base::multiply(npdirs, signals, measurable_size, sig_count);
	if (dst != NULL) {
		data_util::boolD2D(signals, dst, sig_count * measurable_size);
	}
}

vector<vector<vector<bool> > > simulation::abduction(DataManager *dm, const vector<vector<bool> > &signals) {
	std::map<string, int> size_info = dm->getSizeInfo();
	int measurable_size = size_info["_measurable_size"];

	vector<vector<vector<bool> > > results;
	vector<vector<bool> > even_results, odd_results;
	for (int i = 0; i < signals.size(); ++i) {
		vector<int> even_idx, odd_idx;
		for (int j = 0; j < signals[0].size() / 2; ++j) {
			if (signals[i][2 * j]) even_idx.push_back(j);
			if (signals[i][2 * j + 1]) odd_idx.push_back(j);
		}
		if (even_idx.empty()) {
			vector<bool> tmp(measurable_size, false);
			even_results.push_back(tmp);
		}
		else {
			kernel_util::alltrue(dm->_dvar_b(DataManager::SIGNALS), measurable_size);
			for (int j = 0; j < even_idx.size(); ++j) {
				kernel_util::conjunction(dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::NPDIR_MASK) + even_idx[j] * measurable_size, measurable_size);
			}
			vector<vector<bool> > signals = dm->getSignals(1);
			even_results.push_back(signals[0]);
		}
		if (odd_idx.empty()) {
			vector<bool> tmp(measurable_size, false);
			odd_results.push_back(tmp);
		}
		else {
			kernel_util::allfalse(dm->_dvar_b(DataManager::SIGNALS), measurable_size);
			for (int j = 0; j < odd_idx.size(); ++j) {
				kernel_util::disjunction(dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::NPDIR_MASK) + odd_idx[j] * measurable_size, measurable_size);
			}

			kernel_util::allfalse(dm->_dvar_b(DataManager::LOAD), measurable_size);
			simulation::propagates(dm->_dvar_b(DataManager::NPDIRS), dm->_dvar_b(DataManager::LOAD), dm->_dvar_b(DataManager::SIGNALS), dm->_dvar_b(DataManager::LSIGNALS), NULL, 1, measurable_size);
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
	int measurable_size = size_info["_measurable_size"];

	int t = floor(log(sensor_size) / log(2)) + 1;
	for (int i = 0; i < t; ++i) {
		uma_base::dioid_square(dm->_dvar_i(DataManager::DISTS), sensor_size);
	}

	uma_base::union_init(dm->_dvar_i(DataManager::DataManager::UNION_ROOT), sensor_size);
	uma_base::check_dist(dm->_dvar_i(DataManager::DISTS), delta, sensor_size * sensor_size);

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