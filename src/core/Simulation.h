#ifndef _SIMULATION_
#define _SIMULATION_

#include "Global.h"
#include "UMAException.h"

class Agent;
class DataManager;
class Snapshot;

namespace simulation {
	void DLL_PUBLIC update_state(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active);
	void DLL_PUBLIC update_state_qualitative(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active);
	void DLL_PUBLIC propagates(bool *npdirs, bool *load, bool *signals, bool *lsignals, bool *dst, int sig_count, int attr_sensor_size);
	void DLL_PUBLIC floyd(DataManager *dm);
	void DLL_PUBLIC gen_mask(DataManager *dm, int &initial_size);
	void DLL_PUBLIC halucinate(DataManager *dm, int &initial_size);
	void DLL_PUBLIC propagate_mask(DataManager *dm);
	void DLL_PUBLIC calculate_target(DataManager *dm, const int type);
	int DLL_PUBLIC distance(DataManager *dm, bool *b1, bool *b2, int size);
	float DLL_PUBLIC divergence(DataManager *dm);
	void DLL_PUBLIC enrichment(Snapshot *snapshot, bool doPruning);
	float DLL_PUBLIC decide(Snapshot *snapshot, vector<bool> &signal, const double &phi, const bool active);
	vector<float> DLL_PUBLIC decide(Agent *agent, vector<bool> &obs_plus, vector<bool> &obs_minus, const double &phi, const bool &active);
	void DLL_PUBLIC ups_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int attr_sensor_size);
	void DLL_PUBLIC downs_GPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int attr_sensor_size);
	vector<vector<vector<bool> > > DLL_PUBLIC abduction(DataManager *dm, const vector<vector<bool> > &signals);
	vector<vector<int> > DLL_PUBLIC blocks_GPU(DataManager *dm, float delta);
}

#endif
