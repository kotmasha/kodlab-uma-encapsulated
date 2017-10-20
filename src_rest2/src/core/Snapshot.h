#ifndef _SNAPSHOT_
#define _SNAPSHOT_

#include "Global.h"
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

class Measurable;
class Sensor;
class MeasurablePair;
class SensorPair;
class logManager;
class Agent;
class DataManager;
/*
----------------Snapshot Base Class-------------------
*/

/*
This is Snapshot class, it is owed by Agent class.
Snapshot is a base class, based on needs, virtual function can have different implementation
The majority of GPU operation start on the Snapshot object, but this object SHOULD NOT be accessed directly by python code directly
All means of operation on Snapshot should go through Agent class
*/
class Snapshot{
protected:
	double _threshold;
	//threshold value from python
	double _q;
	//q value from python
	int _initial_size;
	//the initial sensor size, initial sensor is the basic sensor without amper/delay, the value is initiated 
	string _uuid;
	//snapshot uuid
	vector<Sensor*> _sensors;
	//all sensor object
	vector<SensorPair*> _sensor_pairs;
	//all sensor pair object
	std::map<string, Sensor*> _sensor_idx;
	//map from sensor id to pointer
	//record current sensor num in the current Snapshot
	int t;
	bool _auto_target;
	bool _propagate_mask;
	logManager *_log;
	string _log_dir;
	DataManager *_dm;
	friend class Agent;

public:
	enum Snapshot_type{STATIONARY, FORGETFUL, UNITTEST};
	//the total sum(a square block sum, wij+wi_j+w_ij+w_i_j), and the value in last iteration
	double _total, _total_;

protected:
	vector<int> convert_list(vector<bool> &list);
	vector<bool> convert_signal_to_sensor(vector<bool> &signal);

public:
	//Snapshot(int type, int base_sensor_size, double threshold, string name, vector<string> sensor_ids, vector<string> sensor_names, bool cal_target, string log_type);
	//Snapshot(ifstream &file, string &log_dir);
	Snapshot(string uuid, string log_dir);

	virtual float decide(vector<bool> &signal, double phi, bool active);

	void add_sensor(std::pair<string, string> &id_pair, vector<double> &diag, vector<vector<double> > &w, vector<vector< bool> > &b);
	void delete_sensor(string &sensor_id);

	/*
	---------------------GET FUNCTION----------------------
	*/
	SensorPair *getSensorPair(Sensor *sensor1, Sensor *sensor2);
	Measurable *getMeasurable(int idx);
	Measurable *getMeasurable(string &measurable_id);
	MeasurablePair *getMeasurablePair(int m_idx1, int m_idx2);
	MeasurablePair *getMeasurablePair(string &mid1, string &mid2);
	vector<bool> getAmperList(string &sensor_id);
	vector<string> getAmperListID(string &sensor_id);
	Sensor *getSensor(string &sensor_id);

	double getTotal();
	double getQ();
	double getThreshold();
	bool getAutoTarget();
	bool getPropagateMask();
	int getInitialSize();

	vector<std::pair<int, pair<string, string> > > getSensorInfo();
	/*
	---------------------GET FUNCTION----------------------
	*/

	/*
	---------------------SET FUNCTION----------------------
	*/
	void setThreshold(double &threshold);
	void setQ(double &q);
	void setAutoTarget(bool &auto_target);
	void setPropagateMask(bool &propagate_mask);
	void setInitialSize(int &initial_size);
	void setInitialSize();
	/*
	---------------------SET FUNCTION----------------------
	*/

	/*
	---------------------COPY FUNCTION---------------------
	before using the copy function, have to make sure all necessary variable are in place
	*/
	//void copy_test_data(Snapshot *snapshot);
	/*
	---------------------COPY FUNCTION---------------------
	*/

	void generate_delayed_weights(int mid, bool merge, std::pair<string, string> &id_pair);
	void ampers(vector<vector<bool> > &lists, vector<std::pair<string, string> > &id_pairs);
	void amper(vector<int> &list, std::pair<string, string> &uuid);
	void delays(vector<vector<bool> > &lists, vector<std::pair<string, string> > &id_pairs);
	void amperand(int mid1, int mid2, bool merge, std::pair<string, string> &id_pair);
	void pruning(vector<bool> &signal);

	bool get_implication(string &sensor1, string &sensor2);
	void create_implication(string &sensor1, string &sensor2);
	void delete_implication(string &sensor1, string &sensor2);

	//void save_snapshot(ofstream &file);

	bool amper_and_signals(Sensor *sensor);

	DataManager *getDM();

	virtual void update_total(double phi, bool active);

	~Snapshot();
};

/*
Stationary Snapshot is not throwing away information when it is not active
*/
class Snapshot_Stationary: public Snapshot{
public:
	//Snapshot_Stationary(ifstream &file, string &log_dir);
	Snapshot_Stationary(string uuid, string log_dir);
	virtual ~Snapshot_Stationary();
	//virtual void update_weights(bool active);
	//virtual void update_thresholds();
	//virtual void orient_all();
	virtual void update_total(double phi, bool active);

protected:
	//double q;
};

/*
Forgetful Snapshot will throw away information when it is not active
*/
/*
class Snapshot_Forgetful: public Snapshot{
public:
	Snapshot_Forgetful(string uuid, string log_dir);
	virtual ~Snapshot_Forgetful();
	virtual void update_weights(bool active);
	virtual void update_thresholds();
	virtual void orient_all();
	virtual void update_total(double phi, bool active);

protected:
	//double q;
};
*/
#endif
