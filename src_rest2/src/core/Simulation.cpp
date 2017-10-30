#include "Simulation.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "data_util.h"
#include "kernel_util.cuh"
#include "uma_base.cuh"

extern int ind(int row, int col);

void simulation::update_total(double &total, double &total_, double &q, double &phi) {
	total_ = total;
	total = q * total + (1 - q) * phi;
}

float simulation::decide(Snapshot *snapshot, vector<bool> &signal, double phi, bool active) {//the decide function
	DataManager *dm = snapshot->getDM();
	double total = snapshot->getTotal();
	double total_ = snapshot->getOldTotal();
	double q = snapshot->getQ();
	int initial_size = snapshot->getInitialSize();

	dm->setObserve(signal);
	simulation::update_total(total, total_, q, phi);
	simulation::update_state(dm, q, phi, total, total_, active);
	simulation::halucinate(dm, initial_size);

	if (snapshot->getPropagateMask()) simulation::propagate_mask(dm);
	if (snapshot->getAutoTarget()) simulation::calculate_target(dm);

	return simulation::divergence(dm);
}

void simulation::update_state(DataManager *dm, double &q, double &phi, double &total, double &total_, bool &active) {
	int sensor_size = dm->getSensorSize();
	int measurable_size = dm->getMeasurableSize();
	//update weights
	uma_base::update_weights(dm->_dvar_d(WEIGHTS), dm->_dvar_b(OBSERVE), measurable_size, q, phi, active);
	//update diag from new weights
	uma_base::get_weights_diag(dm->_dvar_d(WEIGHTS), dm->_dvar_d(DIAG), dm->_dvar_d(OLD_DIAG), measurable_size);
	//update thresholds
	uma_base::update_thresholds(dm->_dvar_b(DIRS), dm->_dvar_d(THRESHOLDS), total_, q, phi, sensor_size);
	//update dirs
	uma_base::orient_all(dm->_dvar_b(DIRS), dm->_dvar_d(WEIGHTS), dm->_dvar_d(WEIGHTS), total, sensor_size);
	//floyd on new dirs and get npdirs
	floyd(dm);

	//the propagation on 
	data_util::boolD2D(dm->_dvar_b(OBSERVE), dm->_dvar_b(SIGNALS), measurable_size);
	kernel_util::allfalse(dm->_dvar_b(LOAD), measurable_size);
	simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(SIGNALS), dm->_dvar_b(LSIGNALS), dm->_dvar_b(CURRENT), 1, measurable_size);
	data_util::boolD2H(dm->_dvar_b(CURRENT), dm->_dvar_b(CURRENT), measurable_size);
}

void simulation::floyd(DataManager *dm) {
	int measurable_size = dm->getMeasurableSize();
	int measurable2d_size = dm->getMeasurable2dSize();
	data_util::boolD2D(dm->_dvar_b(DIRS), dm->_dvar_b(NPDIRS), measurable2d_size);
	uma_base::floyd(dm->_dvar_b(NPDIRS), measurable_size);
	data_util::boolD2H(dm->_hvar_b(NPDIRS), dm->_dvar_b(NPDIRS), measurable2d_size);
	//cudaCheckErrors("kernel fails");
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
}

void simulation::halucinate(DataManager *dm, int &initial_size) {
	int sensor_size = dm->getSensorSize();
	int measurable_size = dm->getMeasurableSize();
	simulation::gen_mask(dm, initial_size);

	data_util::boolD2D(dm->_dvar_b(MASK), dm->_dvar_b(SIGNALS), measurable_size);
	kernel_util::allfalse(dm->_dvar_b(LOAD), measurable_size);
	simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(SIGNALS), dm->_dvar_b(LSIGNALS), dm->_dvar_b(PREDICTION), 1, measurable_size);
	data_util::boolD2H(dm->_dvar_b(PREDICTION), dm->_hvar_b(PREDICTION), measurable_size);
}

void simulation::gen_mask(DataManager *dm,  int &initial_size) {
	int sensor_size = dm->getSensorSize();
	int measurable_size = dm->getMeasurableSize();
	uma_base::init_mask(dm->_dvar_b(MASK), initial_size, measurable_size);

	uma_base::mask(dm->_dvar_b(MASK_AMPER), dm->_dvar_b(MASK), dm->_dvar_b(CURRENT), sensor_size);
	uma_base::check_mask(dm->_dvar_b(MASK), sensor_size);
}

void simulation::propagate_mask(DataManager *dm) {
	int sensor_size = dm->getSensorSize();
	int measurable_size = dm->getMeasurableSize();

	kernel_util::allfalse(dm->_dvar_b(NPDIR_MASK), sensor_size * measurable_size);
	kernel_util::allfalse(dm->_dvar_b(SIGNALS), sensor_size * measurable_size);
	kernel_util::allfalse(dm->_dvar_b(LOAD), measurable_size);
	for (int i = 0; i < sensor_size; ++i) {
		data_util::boolD2D(dm->_dvar_b(MASK_AMPER), dm->_dvar_b(SIGNALS), (ind(i + 1, 0) - ind(i, 0)) * 2, ind(i, 0) * 2, measurable_size * i);
	}
	simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(SIGNALS), dm->_dvar_b(NPDIR_MASK), NULL, sensor_size, measurable_size);
}

void simulation::calculate_target(DataManager *dm) {
	int sensor_size = dm->getSensorSize();
	int measurable_size = dm->getMeasurableSize();

	uma_base::calculate_target(dm->_dvar_d(DIAG), dm->_dvar_b(TARGET), sensor_size);
	data_util::boolD2H(dm->_dvar_b(TARGET), dm->_hvar_b(TARGET), measurable_size);
}


float simulation::divergence(DataManager *dm) {
	int measurable_size = dm->getMeasurableSize();

	kernel_util::allfalse(dm->_dvar_b(LOAD), measurable_size);

	simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(CURRENT), dm->_dvar_b(LSIGNALS), dm->_dvar_b(CURRENT), 1, measurable_size);
	simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(TARGET), dm->_dvar_b(LSIGNALS), dm->_dvar_b(TARGET), 1, measurable_size);

	kernel_util::subtraction(dm->_dvar_b(CURRENT), dm->_dvar_b(TARGET), measurable_size);
	//apply weights to the computed divergence signal and output the result:
	float *tmp_result = new float[1];
	tmp_result[0] = 0.0f;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemcpy(dev_result, tmp_result, sizeof(float), cudaMemcpyHostToDevice);
	//weights are w_{xx}-w_{x*x*}:

	uma_base::delta_weight_sum(dm->_dvar_d(DIAG), dm->_dvar_b(CURRENT), dev_result, measurable_size);
	cudaMemcpy(tmp_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
	float result = tmp_result[0];
	delete[] tmp_result;
	cudaFree(dev_result);
	return result;
}

//still have problem in the function
float simulation::distance(DataManager *dm) {
	int measurable_size = dm->getMeasurableSize();

	kernel_util::allfalse(dm->_dvar_b(LOAD), measurable_size);

	simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(CURRENT), dm->_dvar_b(LSIGNALS), dm->_dvar_b(CURRENT), 1, measurable_size);
	simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(TARGET), dm->_dvar_b(LSIGNALS), dm->_dvar_b(TARGET), 1, measurable_size);

	kernel_util::conjunction_star(dm->_dvar_b(CURRENT), dm->_dvar_b(TARGET), measurable_size);

	//cudaMemcpy(h_signal, signal1, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int sum = 0;
	//for (int i = 0; i < _measurable_size; ++i) sum += h_signal[i];
	return sum;
}

void simulation::ups_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int measurable_size) {
	uma_base::transpose_multiply(npdirs, signals, measurable_size, sig_count);
	if (dst != NULL) {
		data_util::boolD2D(signals, dst, sig_count * measurable_size);
	}
}

vector<vector<vector<bool> > > simulation::abduction(DataManager *dm, vector<vector<bool> > &signals) {
	int measurable_size = dm->getMeasurableSize();

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
			kernel_util::alltrue(dm->_dvar_b(SIGNALS), measurable_size);
			for (int j = 0; j < even_idx.size(); ++j) {
				kernel_util::conjunction(dm->_dvar_b(SIGNALS), dm->_dvar_b(NPDIR_MASK) + even_idx[j] * measurable_size, measurable_size);
			}
			vector<vector<bool> > signals = dm->getSignals(1);
			even_results.push_back(signals[0]);
		}
		if (odd_idx.empty()) {
			vector<bool> tmp(measurable_size, false);
			odd_results.push_back(tmp);
		}
		else {
			kernel_util::allfalse(dm->_dvar_b(SIGNALS), measurable_size);
			for (int j = 0; j < odd_idx.size(); ++j) {
				kernel_util::disjunction(dm->_dvar_b(SIGNALS), dm->_dvar_b(NPDIR_MASK) + odd_idx[j] * measurable_size, measurable_size);
			}

			kernel_util::allfalse(dm->_dvar_b(LOAD), measurable_size);
			simulation::propagates(dm->_dvar_b(NPDIRS), dm->_dvar_b(LOAD), dm->_dvar_b(SIGNALS), dm->_dvar_b(LSIGNALS), NULL, 1, measurable_size);
			vector<vector<bool> > signals = dm->getLSignals(1);
			odd_results.push_back(signals[0]);
		}
	}
	results.push_back(even_results);
	results.push_back(odd_results);
	return results;
}

vector<vector<int> > simulation::blocks_GPU(DataManager *dm, float delta) {
	int sensor_size = dm->getSensorSize();
	int measurable_size = dm->getMeasurableSize();

	int t = floor(log(sensor_size) / log(2)) + 1;
	for (int i = 0; i < t; ++i) {
		uma_base::dioid_square(dm->_dvar_i(DISTS), sensor_size);
	}

	uma_base::union_init(dm->_dvar_i(UNION_ROOT), sensor_size);
	uma_base::check_dist(dm->_dvar_i(DISTS), delta, sensor_size * sensor_size);

	uma_base::union_GPU(dm->_dvar_i(DISTS), dm->_dvar_i(UNION_ROOT), sensor_size);
	data_util::intD2H(dm->_dvar_i(UNION_ROOT), dm->_hvar_i(UNION_ROOT), sensor_size);

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