#ifndef _UMACORETESTFIXTURE_
#define _UMACORETESTFIXTURE_

#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "AttrSensor.h"
#include "AttrSensorPair.h"
#include "DataManager.h"
#include "uma_base.cuh"
#include "gtest/gtest.h"

class WorldTestFixture {
public:
	WorldTestFixture();
	~WorldTestFixture();
};

class AgentTestFixture {

};

class AmperAndTestFixture : public ::testing::Test {
public:
	AmperAndTestFixture();
	~AmperAndTestFixture();

	vector<vector<double>> testAmperAnd(int mid1, int mid2, bool merge, std::pair<string, string> &p);

protected:
	Agent *agent;
	Snapshot *snapshot;
};

class SnapshotUpdateQTestFixture : public::testing::Test {
public:
	SnapshotUpdateQTestFixture();
	~SnapshotUpdateQTestFixture();

	void testUpdateQ();

protected:
	Agent *agentStationary, *agentQualitative, *agentDiscounted, *agentEmpirical;
	Snapshot *snapshotStationary, *snapshotQualitative, *snapshotDiscounted, *snapshotEmpirical;
};

class GenerateDelayedWeightsTestFixture : public::testing::Test {
public:
	GenerateDelayedWeightsTestFixture();
	~GenerateDelayedWeightsTestFixture();

	vector<vector<double>> testGenerateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair, vector<bool> &observe, UMA_AGENT type);

protected:
	Agent *agent, *agentQualitative;
	Snapshot *snapshot, *snapshotQualitative;
};

class AmperTestFixture : public::testing::Test {
public:
	AmperTestFixture();
	~AmperTestFixture();

	vector<vector<double>> testAmper(const vector<int> &list, const std::pair<string, string> &uuid);

protected:
	Agent *agent;
	Snapshot *snapshot;
};

class UMACoreDataFlowTestFixture : public::testing::Test {
public:
	UMACoreDataFlowTestFixture();
	~UMACoreDataFlowTestFixture();

	void testUmaCoreDataFlow(int startIdx, int endIdx);

protected:
	DataManager *dm;
	vector<Sensor *> sensors;
	vector<SensorPair *> sensorPairs;
	vector<double> weights;
	vector<double> thresholds;
	vector<bool> dirs;
	vector<double> diag;
	vector<bool> observe;
};

class SensorValuePointerConvertionTestFixture : public::testing::Test {
public:
	SensorValuePointerConvertionTestFixture();
	~SensorValuePointerConvertionTestFixture();

	void testPointerToNull();
	void testValuesToPointers();
	void testPointersToValues();

protected:
	Sensor *s1, *s2, *s3, *s4;
	vector<Sensor *> s;
	double *h_diag, *h_diag_;
	bool *h_observe, *h_observe_, *h_current, *h_current_, *h_prediction, *h_target;
};

class SensorSavingLoading: public::testing::Test{
public:
	SensorSavingLoading();
	~SensorSavingLoading();
	void savingAndLoading();

protected:
	Sensor *s1, *s2;
	string fileName;
};

class AttrSensorSavingLoading: public::testing::Test {
public:
	AttrSensorSavingLoading();
	~AttrSensorSavingLoading();
	void savingAndLoading();

protected:
	AttrSensor *as1, *as2;
	string fileName;
};

class SensorPairSavingLoading : public::testing::Test{
public:
	SensorPairSavingLoading();
	~SensorPairSavingLoading();
	void savingAndLoading();

protected:
	SensorPair *sp1, *sp2;
	Sensor *s1, *s2;
	string fileName;
};

class AttrSensorPairSavingLoading : public::testing::Test {
public:
	AttrSensorPairSavingLoading();
	~AttrSensorPairSavingLoading();
	void savingAndLoading();

protected:
	AttrSensorPair *asp1, *asp2;
	AttrSensor *as1, *as2;
	string fileName;
};

class SnapshotSavingLoading : public::testing::Test {
public:
	SnapshotSavingLoading();
	~SnapshotSavingLoading();
	void savingAndLoading();

protected:
	Agent *agent1, *agent2;
	Snapshot *snapshot1, *snapshot2;
	string fileName;
};

class AgentSavingLoading : public::testing::Test {
public:
	AgentSavingLoading();
	~AgentSavingLoading();
	void savingAndLoading();

protected:
	Agent *agent1, *agent2;
	string fileName;
};

class ExperimentSavingLoading : public::testing::Test {
public:
	ExperimentSavingLoading();
	~ExperimentSavingLoading();
	void savingAndLoading();

protected:
	Experiment *exp1, *exp2;
};

#endif
