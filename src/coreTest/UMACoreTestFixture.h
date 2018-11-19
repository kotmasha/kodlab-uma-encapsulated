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
	void copying();
	void asserting(Sensor *si, Sensor *sj);

protected:
	Sensor *s1, *s2, *s3;
	string fileName;
};

class AttrSensorSavingLoading: public::testing::Test {
public:
	AttrSensorSavingLoading();
	~AttrSensorSavingLoading();
	void savingAndLoading();
	void copying();
	void asserting(AttrSensor *asi, AttrSensor *asj);

protected:
	AttrSensor *as1, *as2, *as3;
	string fileName;
};

class SensorPairSavingLoading : public::testing::Test{
public:
	SensorPairSavingLoading();
	~SensorPairSavingLoading();
	void savingAndLoading();
	void copying();
	void asserting(SensorPair *spi, SensorPair *spj);

protected:
	SensorPair *sp1, *sp2, *sp3;
	Sensor *s1, *s2;
	string fileName;
};

class AttrSensorPairSavingLoading : public::testing::Test {
public:
	AttrSensorPairSavingLoading();
	~AttrSensorPairSavingLoading();
	void savingAndLoading();
	void copying();
	void asserting(AttrSensorPair *aspi, AttrSensorPair *aspj);

protected:
	AttrSensorPair *asp1, *asp2, *asp3;
	AttrSensor *as1, *as2;
	string fileName;
};

class DataManagerSavingLoading : public::testing::Test {
public:
	DataManagerSavingLoading();
	~DataManagerSavingLoading();
	void savingAndLoading();
	void copying();

protected:
	DataManager *dm1, *dm2, *dm3;
	string fileName;
};

class SnapshotSavingLoading : public::testing::Test {
public:
	SnapshotSavingLoading();
	~SnapshotSavingLoading();
	void savingAndLoading();
	void copying();

protected:
	Agent *agent1, *agent2;
	Snapshot *s1, *s2, *s3, *s4, *s5, *s6, *s7, *s8;
	Snapshot *s9, *s10, *s11, *s12; // for copying tests
	string s1FileName, s2FileName, s3FileName, s4FileName;
};

class AgentSavingLoading : public::testing::Test {
public:
	AgentSavingLoading();
	~AgentSavingLoading();
	void savingAndLoading();
	void copying();

protected:
	Agent *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8;
	Agent *a9, *a10, *a11, *a12; // for copying tests
	string a1FileName, a2FileName, a3FileName, a4FileName;
};

class ExperimentSavingLoading : public::testing::Test {
public:
	ExperimentSavingLoading();
	~ExperimentSavingLoading();
	void savingAndLoading();

protected:
	Experiment *exp1, *exp2;
};

class UMASavingLoading : public::testing::Test {
public:
	UMASavingLoading();
	~UMASavingLoading();
	void savingAndLoading();

protected:
	Experiment *exp1, *exp2;
};

class UMAAgentCopying : public::testing::Test {
public:
	UMAAgentCopying();
	~UMAAgentCopying();
	void copyingAgents();
	void assertingAgents(Agent *agent1, Agent *agent2);

protected:
	Experiment *exp;
};

#endif
