#ifndef _SIMULATION_
#define _SIMULATION_

#include "Global.h"
#include "UMAException.h"

class Agent;
class DataManager;
class Snapshot;

namespace simulation {
	void DLL_PUBLIC update_total(double &total, double &total_, const double &q, const double &phi);
	void DLL_PUBLIC update_state(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active);
	void DLL_PUBLIC propagates(bool *npdirs, bool *load, bool *signals, bool *lsignals, bool *dst, int sig_count, int measurable_size);
	void DLL_PUBLIC floyd(DataManager *dm);
	void DLL_PUBLIC gen_mask(DataManager *dm, int &initial_size);
	void DLL_PUBLIC halucinate(DataManager *dm, int &initial_size);
	void DLL_PUBLIC propagate_mask(DataManager *dm);
	void DLL_PUBLIC calculate_target(DataManager *dm);
	float DLL_PUBLIC divergence(DataManager *dm);
	float DLL_PUBLIC distance(DataManager *dm);
	float DLL_PUBLIC decide(Snapshot *snapshot, const vector<bool> &signal, const double &phi, const bool active);
	vector<float> DLL_PUBLIC decide(Agent *agent, const vector<bool> &obs_plus, const vector<bool> &obs_minus, const double &phi, const bool &active);
	void DLL_PUBLIC ups_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int measurable_size);
	void DLL_PUBLIC downs_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int measurable_size);
	vector<vector<vector<bool> > > DLL_PUBLIC abduction(DataManager *dm, const vector<vector<bool> > &signals);
	vector<vector<int> > DLL_PUBLIC blocks_GPU(DataManager *dm, float delta);
}

#endif
