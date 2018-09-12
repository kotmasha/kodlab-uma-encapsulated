#ifndef _SIMULATION_
#define _SIMULATION_

#include "Global.h"
#include "UMAException.h"

class Agent;
class DataManager;
class Snapshot;

namespace simulation {
	void DLL_PUBLIC updateState(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active);
	void DLL_PUBLIC updateStateQualitative(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active);
	void DLL_PUBLIC updateStateDiscounted(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active);
	void DLL_PUBLIC updateStateEmpirical(DataManager *dm, const double &q, const double &phi, const double &total, const double &total_, const bool &active);
	void DLL_PUBLIC propagates(bool *npdirs, bool *load, bool *signals, bool *lsignals, bool *dst, int sig_count, int attr_sensorSize);
	void DLL_PUBLIC floyd(DataManager *dm);
	void DLL_PUBLIC genMask(DataManager *dm, int &initial_size);
	void DLL_PUBLIC halucinate(DataManager *dm, int &initial_size);
	void DLL_PUBLIC propagateMask(DataManager *dm);
	void DLL_PUBLIC calculateTarget(DataManager *dm, const int type);
	int DLL_PUBLIC distance(DataManager *dm, bool *b1, bool *b2, int size);
	float DLL_PUBLIC divergence(DataManager *dm);
	void DLL_PUBLIC enrichment(Snapshot *snapshot, bool doPruning);
	void DLL_PUBLIC abductionOverNegligible(Snapshot *snapshot, vector<vector<bool>> &sensors_to_be_added, vector<bool> &sensors_to_be_removed);
	void DLL_PUBLIC abductionOverDelayedSensors(Snapshot *snapshot, vector<vector<bool>> &sensors_to_be_added, vector<bool> &sensors_to_be_removed);
	float DLL_PUBLIC decide(Snapshot *snapshot, vector<bool> &signal, const double &phi, const bool active);
	vector<float> DLL_PUBLIC decide(Agent *agent, vector<bool> &obs_plus, vector<bool> &obs_minus, const double &phi, const bool &active);
	void DLL_PUBLIC upsGPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int attr_sensorSize);
	void DLL_PUBLIC downsGPU(bool *npdirs, bool *signals, bool *dst, int sig_count, int attr_sensorSize);
	vector<vector<vector<bool> > > DLL_PUBLIC abduction(DataManager *dm, const vector<vector<bool> > &signals);
	vector<vector<int> > DLL_PUBLIC blocksGPU(DataManager *dm, float delta);
}

#endif
