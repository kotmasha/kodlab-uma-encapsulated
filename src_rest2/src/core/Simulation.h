#ifndef _SIMULATION_
#define _SIMULATION_

#include "Global.h"
#include "UMAException.h"

class DataManager;
class Snapshot;

namespace simulation {
	void update_total(double &total, double &total_, double &q, double &phi);
	void update_state(DataManager *dm, double &q, double &phi, double &total, double &total_, bool &active);
	void propagates(bool *npdirs, bool *load, bool *signals, bool *lsignals, bool *dst, int sig_count, int measurable_size);
	void floyd(DataManager *dm);
	void gen_mask(DataManager *dm, int &initial_size);
	void halucinate(DataManager *dm, int &initial_size);
	void propagate_mask(DataManager *dm);
	void calculate_target(DataManager *dm);
	float divergence(DataManager *dm);
	float distance(DataManager *dm);
	float decide(Snapshot *snapshot, vector<bool> &signal, double phi, bool active);
	void ups_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int measurable_size);
	vector<vector<vector<bool> > > abduction(DataManager *dm, vector<vector<bool> > &signals);
	vector<vector<int> > blocks_GPU(DataManager *dm, float delta);
}

#endif
