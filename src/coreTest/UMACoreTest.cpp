#include <iostream>
#include "gtest/gtest.h"
#include "UMACoreTestFixture.h"
#include "UMAException.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "AttrSensor.h"
#include "AttrSensorPair.h"
#include "UMACoreConstant.h"
#include "CoreService.h"
#include "PropertyMap.h"
#include "PropertyPage.h"

TEST(world_test, world_experiment_test) {
	// testing reload world first
	World::instance();
	World::reset();

	//test experiment create
	Experiment *ex1 = World::instance()->createExperiment("ex1");
	Experiment *ex2 = World::instance()->createExperiment("ex2");
	Experiment *ex3 = World::instance()->createExperiment("ex3");
	Experiment *ex4 = World::instance()->createExperiment("ex4");

	EXPECT_NO_THROW(World::instance()->getExperiment("ex2"));
	EXPECT_THROW(World::instance()->getExperiment("ex5"), UMANoResourceException);

	//test experiment get
	vector<string> s = { "ex1", "ex2", "ex3", "ex4" };
	EXPECT_EQ(s, World::instance()->getExperimentInfo());

	//test experiment delete
	EXPECT_NO_THROW(World::instance()->deleteExperiment("ex1"));
	EXPECT_THROW(World::instance()->deleteExperiment("ex5"), UMANoResourceException);

	s = { "ex2", "ex3", "ex4" };
	EXPECT_EQ(s, World::instance()->getExperimentInfo());

	//delete all experiments
	World::instance()->deleteExperiment("ex2");
	World::instance()->deleteExperiment("ex3");
	World::instance()->deleteExperiment("ex4");
}

TEST(experiment_test, experiment_agent_test) {
	//test agent create
	Experiment *experiment = World::instance()->createExperiment("testExperiment");
	experiment->createAgent("testAgent1", UMA_AGENT::AGENT_STATIONARY);
	experiment->createAgent("testAgent2", UMA_AGENT::AGENT_QUALITATIVE);
	experiment->createAgent("testAgent3", UMA_AGENT::AGENT_DISCOUNTED);
	experiment->createAgent("testAgent4", UMA_AGENT::AGENT_EMPIRICAL);

	EXPECT_NO_THROW(experiment->getAgent("testAgent1"));
	EXPECT_THROW(experiment->getAgent("testAgent0"), UMANoResourceException);
	//test agent get
	vector<vector<string>> s = { { "testAgent1", "Stationary" },{ "testAgent2", "Qualitative" },{ "testAgent3", "Discounted" },{ "testAgent4", "Empirical" } };
	EXPECT_EQ(s, experiment->getAgentInfo());

	//test agent delete
	experiment->deleteAgent("testAgent1");
	s = { { "testAgent2", "Qualitative" },{ "testAgent3", "Discounted" },{ "testAgent4", "Empirical" } };
	EXPECT_EQ(s, experiment->getAgentInfo());

	experiment->deleteAgent("testAgent3");
	s = { { "testAgent2", "Qualitative" },{ "testAgent4", "Empirical" } };
	EXPECT_EQ(s, experiment->getAgentInfo());

	EXPECT_THROW(experiment->getAgent("testAgent3"), UMANoResourceException);
	EXPECT_THROW(experiment->deleteAgent("testAgent3"), UMANoResourceException);
	
	experiment->deleteAgent("testAgent2");
	experiment->deleteAgent("testAgent4");
	World::instance()->deleteExperiment("testExperiment");
}

TEST_F(ExperimentSavingLoading, experiment_save_load) {
	savingAndLoading();
}

TEST(agent_test, agent_snapshot_test) {
	vector<Agent*> agents;
	Agent *agentStationary = new Agent("agent", nullptr);
	Agent *agentQualitative = new AgentQualitative("agentQualitative", nullptr);
	Agent *agentDiscounted = new AgentDiscounted("agentDiscounted", nullptr);
	Agent *agentEmpirical = new AgentEmpirical("agentEmpirical", nullptr);
	agents.push_back(agentStationary);
	agents.push_back(agentQualitative);
	agents.push_back(agentDiscounted);
	agents.push_back(agentEmpirical);
	vector<string> agentType = { "Stationary", "Qualitative", "Discounted", "Empirical" };
	vector<string> snapshotType = agentType;

	for (int i = 0; i < agentType.size(); ++i) {
		Agent *agent = agents[i];
		string type = agentType[i];
		Snapshot *s1 = agent->createSnapshot("snapshot1");
		Snapshot *s2 = agent->createSnapshot("snapshot2");
		Snapshot *s3 = agent->createSnapshot("snapshot3");
		Snapshot *s4 = agent->createSnapshot("snapshot4");
		EXPECT_EQ(UMACoreConstant::getUMAAgentName(agent->getType()), agentType[i]);
		EXPECT_EQ(UMACoreConstant::getUMASnapshotName(s1->getType()), snapshotType[i]);

		EXPECT_NO_THROW(agent->getSnapshot("snapshot2"));
		EXPECT_THROW(agent->getSnapshot("snapshot0"), UMANoResourceException);
		vector<vector<string>> s = { { "snapshot1", type },{ "snapshot2", type },{ "snapshot3", type },{ "snapshot4", type } };
		EXPECT_EQ(agent->getSnapshotInfo(), s);

		agent->deleteSnapshot("snapshot2");
		s = { { "snapshot1", type },{ "snapshot3", type },{ "snapshot4", type } };
		EXPECT_EQ(agent->getSnapshotInfo(), s);

		agent->deleteSnapshot("snapshot4");
		s = { { "snapshot1", type },{ "snapshot3", type } };
		EXPECT_EQ(agent->getSnapshotInfo(), s);

		EXPECT_THROW(agent->getSnapshot("snapshot2"), UMANoResourceException);
		EXPECT_THROW(agent->deleteSnapshot("snapshot2"), UMANoResourceException);

		delete agent;
	}
}

TEST(agent_test, get_set_test) {
	Agent *agent = new Agent("agent", nullptr);

	EXPECT_EQ(agent->getT(), 0);
	agent->setT(100);
	EXPECT_EQ(agent->getT(), 100);

	agent->setEnableEnrichment(1);
	EXPECT_EQ(agent->getEnableEnrichment(), true);

	agent->setEnableEnrichment(0);
	EXPECT_EQ(agent->getEnableEnrichment(), false);

	agent->setPruningInterval(100);
	EXPECT_EQ(agent->getPruningInterval(), 100);

	agent->setPruningInterval(5);
	EXPECT_EQ(agent->getPruningInterval(), 5);

	delete agent;
}

TEST(agent_test, do_pruning_test) {
	Agent *agent = new Agent("agent", nullptr);

	agent->setPruningInterval(3);

	agent->setT(agent->getT() + 1);
	EXPECT_EQ(agent->doPruning(), false);
	
	agent->setT(agent->getT() + 1);
	EXPECT_EQ(agent->doPruning(), false);

	agent->setT(agent->getT() + 1);
	EXPECT_EQ(agent->doPruning(), true);

	delete agent;
}

TEST_F(AgentSavingLoading, agent_save_load) {
	savingAndLoading();
	copying();
}

TEST(snapshot_test, snapshotCreateSensorTest) {
	Agent *agent = new Agent("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");
	snapshot->setTotal(1);
	snapshot->setOldTotal(1);
	snapshot->setThreshold(0.125);

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> diag;
	vector<vector<double>> w;
	vector<vector<bool>> b;
	//test sensor create
	snapshot->createSensor(sensor1, diag, w, b);
	snapshot->createSensor(sensor2, diag, w, b);
	snapshot->createSensor(sensor3, diag, w, b);
	snapshot->createSensor(sensor4, diag, w, b);

	//test sensor get
	EXPECT_NO_THROW(snapshot->getSensor("s1"));
	EXPECT_NO_THROW(snapshot->getSensor("cs2"));
	EXPECT_THROW(snapshot->getSensor("s0"), UMANoResourceException);

	//test sensor delete
	snapshot->deleteSensor("s1");
	vector<vector<string>> s = { {"s2", "cs2"}, {"s3", "cs3"}, {"s4", "cs4"} };
	EXPECT_EQ(s, snapshot->getSensorInfo());
	snapshot->deleteSensor("cs3");
	s = { { "s2", "cs2" }, { "s4", "cs4" } };
	EXPECT_EQ(s, snapshot->getSensorInfo());

	//test sensor  get
	EXPECT_THROW(snapshot->getSensor("s1"), UMANoResourceException);
	EXPECT_THROW(snapshot->getSensor("cs3"), UMANoResourceException);
	EXPECT_THROW(snapshot->deleteSensor("s1"), UMANoResourceException);
	EXPECT_THROW(snapshot->deleteSensor("cs3"), UMANoResourceException);

	EXPECT_DOUBLE_EQ(snapshot->getAttrSensor("s2")->getDiag(), 0.5);

	diag = { 0.32, 0.68 };
	std::pair<string, string> sensor5 = { "s5", "cs5" };
	snapshot->createSensor(sensor5, diag, w, b);
	EXPECT_DOUBLE_EQ(snapshot->getAttrSensor("s5")->getDiag(), 0.32);
	EXPECT_DOUBLE_EQ(snapshot->getAttrSensor("cs5")->getDiag(), 0.68);

	delete agent;
}

TEST(snapshot_test, snapshotCreateSensorPairTest) {
	Agent *agent = new Agent("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");
	snapshot->setTotal(1);
	snapshot->setOldTotal(1);
	snapshot->setThreshold(0.125);

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> diag;
	vector<vector<double>> w1, w3;
	vector<vector<double>> w0 = { {0.2, 0.8, 0.457, 0.543} };
	vector<vector<double>> w2 = { {0.2, 0.8, 0.4, 0.6}, {0.1, 0.9, 0, 1}, {0.3, 0.7, 0.8, 0.2} };
	vector<vector<bool>> b1, b3;
	vector<vector<bool>> b0 = { {true, false, true, false} };
	vector<vector<bool>> b2 = { { true, false, false, true }, {false, false, false, false}, {true, true, true, true} };
	// test sensor pair create
	Sensor *s1 = snapshot->createSensor(sensor1, diag, w0, b0);
	Sensor *s2 = snapshot->createSensor(sensor2, diag, w1, b1);
	Sensor *s3 = snapshot->createSensor(sensor3, diag, w2, b2);
	snapshot->setThreshold(0.501);
	Sensor *s4 = snapshot->createSensor(sensor4, diag, w3, b3);

	// test sensor pair get
	EXPECT_NO_THROW(snapshot->getSensorPair(snapshot->getSensor("s1"), snapshot->getSensor("cs3")));
	EXPECT_THROW(snapshot->getSensorPair(snapshot->getSensor("s0"), snapshot->getSensor("cs3")), UMAException);

	EXPECT_EQ(snapshot->getSensorPair(s1, s3), snapshot->getSensorPair(s3, s1));
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(true, true)->getW(), 0.2);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(true, false)->getW(), 0.457);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(false, true)->getW(), 0.457);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(false, false)->getW(), 0.543);

	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s2, s3)->getAttrSensorPair(true, true)->getW(), 0.1);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s3, s2)->getAttrSensorPair(true, false)->getW(), 0.9);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s3, s2)->getAttrSensorPair(false, true)->getW(), 0.0);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s2, s3)->getAttrSensorPair(false, false)->getW(), 1.0);

	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s4, s2)->getAttrSensorPair(true, true)->getW(), 0.25);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s2, s4)->getAttrSensorPair(true, false)->getW(), 0.25);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s4, s2)->getAttrSensorPair(false, true)->getW(), 0.25);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s2, s4)->getAttrSensorPair(false, false)->getW(), 0.25);

	EXPECT_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(true, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(true, false)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(false, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(false, false)->getD(), false);

	EXPECT_EQ(snapshot->getSensorPair(s3, s3)->getAttrSensorPair(true, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s3, s3)->getAttrSensorPair(true, false)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s3, s3)->getAttrSensorPair(false, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s3, s3)->getAttrSensorPair(false, false)->getD(), true);

	EXPECT_EQ(snapshot->getSensorPair(s4, s4)->getAttrSensorPair(true, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s4, s4)->getAttrSensorPair(true, false)->getD(), false);
	EXPECT_EQ(snapshot->getSensorPair(s4, s4)->getAttrSensorPair(false, true)->getD(), false);
	EXPECT_EQ(snapshot->getSensorPair(s4, s4)->getAttrSensorPair(false, false)->getD(), true);

	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getThreshold(), 0.125);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s4, s1)->getThreshold(), 0.501);

	// test sensor pair delete
	snapshot->deleteSensor("s2");
	EXPECT_THROW(snapshot->getSensorPair(snapshot->getSensor("s2"), snapshot->getSensor("s4")), UMANoResourceException);
	EXPECT_NO_THROW(snapshot->getSensorPair(snapshot->getSensor("s1"), snapshot->getSensor("s3")));

	delete agent;
}

TEST(snapshot_test, snapshotGetSetAttributeTest) {
	Agent *agent = new Agent("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");

	vector<double> diag;
	vector<vector<double>> w;
	vector<vector<bool>> b;
	std::pair<string, string> s1 = { "s1", "cs1" };
	std::pair<string, string> s2 = { "s2", "cs2" };
	snapshot->createSensor(s1, diag, w, b);
	snapshot->createSensor(s2, diag, w, b);

	snapshot->setThreshold(0.501);
	snapshot->setTotal(2.0);
	snapshot->setOldTotal(0.31);
	snapshot->setAutoTarget(true);
	snapshot->setPropagateMask(true);
	snapshot->setQ(0.8);
	snapshot->setInitialSize();

	EXPECT_DOUBLE_EQ(snapshot->getThreshold(), 0.501);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 2.0);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 0.31);
	EXPECT_EQ(snapshot->getAutoTarget(), true);
	EXPECT_EQ(snapshot->getPropagateMask(), true);
	EXPECT_DOUBLE_EQ(snapshot->getQ(), 0.8);
	EXPECT_EQ(snapshot->getInitialSize(), 2);
	EXPECT_EQ(snapshot->getDelayCount(), 0);
	EXPECT_EQ(snapshot->getDelayCount(), 1);

	snapshot->setInitialSize(5);
	EXPECT_EQ(snapshot->getInitialSize(), 5);
	snapshot->updateTotal(1.1, true);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.82);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 2.0);

	snapshot->updateTotal(1.2, false);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.82);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 1.82);

	delete agent;
}

TEST(snapshot_qualitative_test, updateTotal) {
	Agent *agent = new AgentQualitative("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");
	snapshot->setTotal(1);
	snapshot->setOldTotal(1);
	snapshot->setThreshold(0.125);

	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.0);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 1.0);

	snapshot->setTotal(2.0);
	snapshot->setOldTotal(0.31);

	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 2.0);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 0.31);

	snapshot->updateTotal(1.1, true);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.1);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 2.0);

	snapshot->updateTotal(1.2, false);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.1);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 1.1);

	snapshot->updateTotal(1.2, true);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.1);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 1.1);

	delete agent;
}

TEST_F(SnapshotUpdateQTestFixture, updateQ_test) {
	testUpdateQ();
}

TEST(snapshot_test, snapshotGetEntity) {
	Agent *agent = new Agent("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");
	
	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> diag;
	vector<vector<double>> w;
	vector<vector<bool>> b;
	Sensor *s1 = snapshot->createSensor(sensor1, diag, w, b);
	Sensor *s2 = snapshot->createSensor(sensor2, diag, w, b);
	Sensor *s3 = snapshot->createSensor(sensor3, diag, w, b);
	Sensor *s4 = snapshot->createSensor(sensor4, diag, w, b);

	EXPECT_EQ(s1, snapshot->getSensor("s1"));
	EXPECT_EQ(s2, snapshot->getSensor("cs2"));
	EXPECT_NE(s3, snapshot->getSensor("s4"));

	EXPECT_EQ(snapshot->getAttrSensor("cs3"), snapshot->getAttrSensor(5));
	EXPECT_EQ(snapshot->getAttrSensor(2), snapshot->getAttrSensor("s2"));
	EXPECT_NE(snapshot->getAttrSensor("s3"), snapshot->getAttrSensor("cs3"));
	EXPECT_NE(snapshot->getAttrSensor(0), snapshot->getAttrSensor(1));

	EXPECT_EQ(snapshot->getAttrSensorPair("s3", "cs2"), snapshot->getAttrSensorPair(3, 4));
	EXPECT_EQ(snapshot->getAttrSensorPair("s1", "cs4"), snapshot->getAttrSensorPair(7, 0));
	EXPECT_NE(snapshot->getAttrSensorPair("s4", "s3"), snapshot->getAttrSensorPair(0, 1));

	delete agent;
}

TEST(snapshot_test, generateDelayedObservations) {
	Agent *agent = new Agent("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<std::pair<string, string>> sensor5_6 = { { "s5", "cs5" }, {"s6", "cs6"} };
	vector<std::pair<string, string>> sensor7 = { { "s7", "cs7" } };
	vector<double> diag;
	vector<vector<double>> w;
	vector<vector<bool>> b;
	Sensor *s1 = snapshot->createSensor(sensor1, diag, w, b);
	Sensor *s2 = snapshot->createSensor(sensor2, diag, w, b);
	Sensor *s3 = snapshot->createSensor(sensor3, diag, w, b);
	Sensor *s4 = snapshot->createSensor(sensor4, diag, w, b);
	snapshot->setInitialSize();

	vector<vector<bool>> list1 = { { true, false, false, true, false, true, true, false },{ false, true, false, false, false, true, false, false, false, false } };
	vector<vector<bool>> list2 = { { false, true, false, true, true, false, true, false, true, false, false, false } };
	snapshot->delays(list1, sensor5_6);
	snapshot->ampers(list2, sensor7);
	vector<bool> observe = { false, true, false, true, false, true, true, false};
	//1st set, h_observe get the value, h_observe_ still empty
	snapshot->generateObserve(observe);
	for (int i = 0; i < 6; ++i) observe.pop_back();
	//2nd set, h_observe_ get the value, but after delay calculation(delay still calculation on empty h_observe_)
	snapshot->generateObserve(observe);
	for (int i = 0; i < 6; ++i) observe.pop_back();
	//3rd set, delay get correct h_observe_ to calculate
	snapshot->generateObserve(observe);
	for (int i = 0; i < 6; ++i) observe.pop_back();
	EXPECT_EQ(false, snapshot->getSensor("s5")->getObserve());
	EXPECT_EQ(true, snapshot->getSensor("cs6")->getObserve());
	EXPECT_EQ(false, snapshot->getSensor("s7")->getObserve());

	delete agent;
}

TEST(snapshot_test, generateSignal) {
	Agent *agent = new Agent("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> diag;
	vector<vector<double>> w;
	vector<vector<bool>> b;
	Sensor *s1 = snapshot->createSensor(sensor1, diag, w, b);
	Sensor *s2 = snapshot->createSensor(sensor2, diag, w, b);
	Sensor *s3 = snapshot->createSensor(sensor3, diag, w, b);
	Sensor *s4 = snapshot->createSensor(sensor4, diag, w, b);

	vector<AttrSensor*> m;
	m.push_back(snapshot->getAttrSensor(0));
	m.push_back(snapshot->getAttrSensor(3));
	m.push_back(snapshot->getAttrSensor(6));
	vector<bool> v1 = snapshot->generateSignal(m);
	vector<bool> v2 = { 1, 0, 0, 1, 0, 0, 1, 0 };

	EXPECT_EQ(v1, v2);

	delete agent;
}

TEST_F(SnapshotSavingLoading, snapshot_save_load) {
	savingAndLoading();
	copying();
}

TEST_F(AmperAndTestFixture, snapshot_amper_and_test1) {
	std::pair<string, string> p = {"s5", "cs5"};
	vector<vector<double>> target = testAmperAnd(0, 2, true, p);

	vector<vector<double>> w = {
		{0.2},
		{0.0, 0.8},
		{0.2, 0.2, 0.4},
		{0.0, 0.6, 0.0, 0.6},
		{0.2, 0.4, 0.4, 0.2, 0.6},
		{0.0, 0.4, 0.0, 0.4, 0.0, 0.4},
		{0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8},
		{0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2},
		{0.2, 0.0, 0.2, 0.0, 0.12, 0.08, 0.16, 0.04, 0.2},
		{0.0, 0.8, 0.2, 0.6, 0.48, 0.32, 0.64, 0.16, 0.0, 0.8}
	};
	
	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}

	target = testAmperAnd(4, 8, false, p);
	p = { "s6", "cs6" };

	w = {
		{ 0.2 },
		{ 0.0, 0.8 },
		{ 0.2, 0.2, 0.4 },
		{ 0.0, 0.6, 0.0, 0.6 },
		{ 0.2, 0.4, 0.4, 0.2, 0.6 },
		{ 0.0, 0.4, 0.0, 0.4, 0.0, 0.4 },
		{ 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8 },
		{ 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2 },
		{ 0.024, 0.096, 0.048, 0.072, 0.12, 0.0, 0.096, 0.024, 0.12 },
		{ 0.176, 0.704, 0.352, 0.528, 0.48, 0.4, 0.704, 0.176, 0.0, 0.88 }
	};
	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}
}

TEST_F(GenerateDelayedWeightsTestFixture, snapshot_generateDelayedWeights_test1) {
	std::pair<string, string> p = { "s5", "cs5" };
	vector<bool> observe = { true, false, false, false, false, false, false, false };
	vector<vector<double>> target = testGenerateDelayedWeights(0, true, p, observe, UMA_AGENT::AGENT_STATIONARY);

	vector<vector<double>> w = {
		{ 0.2 },
		{ 0.0, 0.8 },
		{ 0.2, 0.2, 0.4 },
		{ 0.0, 0.6, 0.0, 0.6 },
		{ 0.2, 0.4, 0.4, 0.2, 0.6 },
		{ 0.0, 0.4, 0.0, 0.4, 0.0, 0.4 },
		{ 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8 },
		{ 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2 },
		{ 0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.8, 0.2, 1.0 },
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
	};

	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}
}

TEST_F(GenerateDelayedWeightsTestFixture, snapshot_generateDelayedWeights_test2) {
	std::pair<string, string> p = { "s5", "cs5" };
	vector<bool> observe = { false, true, false, false, false, false, false, false };
	vector<vector<double>> target = testGenerateDelayedWeights(0, true, p, observe, UMA_AGENT::AGENT_STATIONARY);

	vector<vector<double>> w = {
		{ 0.2 },
		{ 0.0, 0.8 },
		{ 0.2, 0.2, 0.4 },
		{ 0.0, 0.6, 0.0, 0.6 },
		{ 0.2, 0.4, 0.4, 0.2, 0.6 },
		{ 0.0, 0.4, 0.0, 0.4, 0.0, 0.4 },
		{ 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8 },
		{ 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2 },
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{ 0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.8, 0.2, 0.0, 1.0 }
	};

	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}
}

TEST_F(GenerateDelayedWeightsTestFixture, snapshot_generateDelayedWeights_test3) {
	std::pair<string, string> p = { "s5", "cs5" };
	vector<bool> observe = { true, false, false, false, false, false, false, false };
	vector<vector<double>> target = testGenerateDelayedWeights(0, true, p, observe, UMA_AGENT::AGENT_QUALITATIVE);

	for (int i = 0; i < target.size(); ++i) {
		for (int j = 0; j < target[i].size(); ++j)
			EXPECT_DOUBLE_EQ(-1, target[i][j]);
	}
}

TEST_F(AmperTestFixture, amper_test) {
	std::pair<string, string> p = { "s5", "cs5" };
	vector<int> list = { 3, 5, 7 };

	vector<vector<double>> target = testAmper(list, p);

	vector<vector<double>> w = {
		{ 0.2 },
		{ 0.0, 0.8 },
		{ 0.2, 0.2, 0.4 },
		{ 0.0, 0.6, 0.0, 0.6 },
		{ 0.2, 0.4, 0.4, 0.2, 0.6 },
		{ 0.0, 0.4, 0.0, 0.4, 0.0, 0.4 },
		{ 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8 },
		{ 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2 },
		{ 0.016, 0.064, 0.032, 0.048, 0.048, 0.032, 0.0, 0.08, 0.08 },
		{ 0.184, 0.736, 0.368, 0.552, 0.552, 0.368, 0.8, 0.12, 0.0, 0.92 }
	};

	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}
}

TEST(dataManager_test, get_set_test) {
	Agent *agent = new Agent("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");

	snapshot->setTotal(1);
	snapshot->setOldTotal(1);
	snapshot->setThreshold(0.125);

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	snapshot->setInitialSize(3);
	vector<double> diag;
	vector<double> diag1 = { 0.2, 0.8 };
	vector<vector<double>> w1, w3;
	vector<vector<double>> w0 = { { 0.2, 0.8, 0.457, 0.543 } };
	vector<vector<double>> w2 = { { 0.2, 0.8, 0.4, 0.6 },{ 0.1, 0.9, 0, 1 },{ 0.3, 0.7, 0.8, 0.2 } };
	vector<vector<bool>> b1, b3;
	vector<vector<bool>> b0 = { { true, false, true, false } };
	vector<vector<bool>> b2 = { { true, false, false, true },{ false, false, false, false },{ true, true, true, true } };
	Sensor *s1 = snapshot->createSensor(sensor1, diag, w0, b0);
	Sensor *s2 = snapshot->createSensor(sensor2, diag1, w1, b1);
	Sensor *s3 = snapshot->createSensor(sensor3, diag, w2, b2);
	snapshot->setThreshold(0.501);
	Sensor *s4 = snapshot->createSensor(sensor4, diag, w3, b3);

	DataManager *dm = snapshot->getDM();
	//mask test
	vector<bool> mask = { false, false, false, true, true, true };
	EXPECT_THROW(dm->setMask(mask), UMAException);
	mask.push_back(false); mask.push_back(false);
	dm->setMask(mask);
	EXPECT_EQ(mask, dm->getMask());
	//load test
	vector<bool> load = { true, false, false, true, false, false };
	EXPECT_THROW(dm->setLoad(load), UMAException);
	load.push_back(false); load.push_back(false);
	dm->setLoad(load);
	EXPECT_EQ(load, dm->getLoad());
	//signals test
	vector<vector<bool>> signals = {
		{true},{false},{true},{false},{true}
	};
	EXPECT_THROW(dm->setSignals(signals), UMAException);
	signals = { {true, false, true, false, false, false},
	{false, true, false, true, true, true, false, false}
	};
	EXPECT_THROW(dm->setSignals(signals), UMAException);
	signals[0].push_back(false); signals[0].push_back(true);
	dm->setSignals(signals);
	EXPECT_EQ(signals, dm->getSignals(2));
	//lsignals test
	vector<vector<bool>> lsignals = {
		{ true },{ false },{ true },{ false },{ true }
	};
	dm->setLoad(load);
	EXPECT_THROW(dm->setLSignals(lsignals), UMAException);
	lsignals = { { true, false, true, false, false, false },
	{ false, true, false, true, true, true, false, false }
	};
	EXPECT_THROW(dm->setLSignals(lsignals), UMAException);
	lsignals[0].push_back(false); lsignals[0].push_back(true);
	dm->setLSignals(lsignals);
	lsignals = { { true, false, true, true, false, false, false, true },
	{ true, true, false, true, true, true, false, false }
	};
	EXPECT_EQ(lsignals, dm->getLSignals(2));
	//dists test
	vector<vector<int>> dists = { {1}, {2}, {3} };
	EXPECT_THROW(dm->setDists(dists), UMAException);
	dists = {
		{1, 1, 1},
		{2, 2, 2, 2},
		{3, 3, 3, 3},
		{4, 4, 4, 4}
	};
	EXPECT_THROW(dm->setDists(dists), UMAException);
	dists[0].push_back(1);
	EXPECT_NO_THROW(dm->setDists(dists));
	//current test
	vector<bool> current = { true, false, false, true, true, true };
	EXPECT_THROW(dm->setCurrent(current), UMAException);
	current.push_back(false); current.push_back(false);
	dm->setCurrent(current);
	EXPECT_EQ(current, dm->getCurrent());
	//old current test
	vector<bool> old_current = { true, false, false, true, true, true };
	EXPECT_THROW(dm->setOldCurrent(old_current), UMAException);
	old_current.push_back(false); old_current.push_back(false);
	dm->setOldCurrent(old_current);
	EXPECT_EQ(old_current, dm->getOldCurrent());
	//target test
	vector<bool> target = { false, true, false, true, true, true };
	EXPECT_THROW(dm->setTarget(target), UMAException);
	target.push_back(false); target.push_back(true);
	dm->setTarget(target);
	EXPECT_EQ(target, dm->getTarget());
	//observe test
	vector<bool> observe = { true, false, false, true, true, false};
	EXPECT_THROW(dm->setObserve(observe), UMAException);
	observe.push_back(false); observe.push_back(false);
	EXPECT_NO_THROW(dm->setObserve(observe));
	EXPECT_EQ(observe, dm->getObserve());

	//weights test
	vector<vector<double>> t_weight2d = {
		{0.2},
		{0.457, 0.543},
		{0.25, 0.25, 0.25},
		{0.25, 0.25, 0.25, 0.25},
		{0.2, 0.8, 0.1, 0.9, 0.3},
		{0.4, 0.6, 0.0, 1.0, 0.8, 0.2},
		{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 },
		{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 },
	};
	vector<double> t_weight1d = { 0.2, 0.457, 0.543, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
	0.2, 0.8, 0.1, 0.9, 0.3, 0.4, 0.6, 0.0, 1.0, 0.8, 0.2, 0.25, 0.25, 0.25, 0.25,0.25,
	0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 };
	vector<vector<double>> r_weight2d = dm->getWeight2D();
	vector<double> r_weight1d = dm->getWeight();
	for (int i = 0; i < t_weight2d.size(); ++i) {
		for (int j = 0; j < t_weight2d[i].size(); ++j)
			EXPECT_DOUBLE_EQ(t_weight2d[i][j], r_weight2d[i][j]);
	}
	for (int i = 0; i < t_weight1d.size(); ++i) {
		EXPECT_DOUBLE_EQ(t_weight1d[i], r_weight1d[i]);
	}

	//dirs test
	vector<vector<bool>> t_dirs2d = {
		{ true },
		{ true, false },
		{ false, false, true },
		{ false, false, false, true },
		{ true, false, false, false, true },
		{ false, true, false, false, true, true },
		{ false, false, false, false, false, false, true },
		{ false, false, false, false, false, false, false, true },
	};
	vector<bool> t_dirs1d = { true, true, false, false, false, true, false, false, false, true,
		true, false, false, false, true, false, true, false, false, true, true, false, false, false, false,
		false, false, true, false, false, false, false, false, false, false, true };
	vector<vector<bool>> r_dirs2d = dm->getDir2D();
	vector<bool> r_dirs1d = dm->getDir();
	EXPECT_EQ(t_dirs2d, r_dirs2d);
	EXPECT_EQ(t_dirs1d, r_dirs1d);

	//threshold test
	vector<vector<double>> t_threshold2d = {
		{0.125},
		{0.125, 0.125},
		{0.125, 0.125, 0.125},
		{0.501, 0.501, 0.501, 0.501}
	};
	vector<double> t_threshold1d = { 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.501, 0.501, 0.501, 0.501 };
	vector<vector<double>> r_threshold2d = dm->getThreshold2D();
	for (int i = 0; i < t_threshold2d.size(); ++i) {
		for (int j = 0; j < t_threshold2d[i].size(); ++j)
			EXPECT_DOUBLE_EQ(r_threshold2d[i][j], t_threshold2d[i][j]);
	}

	vector<double> r_threshold1d = dm->getThresholds();
	for (int i = 0; i < r_threshold1d.size(); ++i)
		EXPECT_DOUBLE_EQ(r_threshold1d[i], t_threshold1d[i]);

	//diag test
	vector<double> t_diag_ = { 0.5, 0.5, 0.2, 0.8, 0.5, 0.5 };
	vector<double> r_diag_ = dm->getDiagOld();
	for (int i = 0; i < t_diag_.size(); ++i)
		EXPECT_DOUBLE_EQ(t_diag_[i], r_diag_[i]);

	vector<double> t_diag = {0.5, 0.5, 0.2, 0.8, 0.5, 0.5};
	vector<double> r_diag = dm->getDiag();
	for (int i = 0; i < t_diag.size(); ++i)
		EXPECT_DOUBLE_EQ(t_diag[i], r_diag[i]);

	delete agent;
}

TEST_F(DataManagerSavingLoading, dm_save_load) {
	savingAndLoading();
	copying();
}

TEST_F(UMACoreDataFlowTestFixture, uma_core_dataflow_test1) {
	EXPECT_THROW(testUmaCoreDataFlow(0, 5), UMAException);
}

TEST_F(UMACoreDataFlowTestFixture, uma_core_dataflow_test2) {
	testUmaCoreDataFlow(0, 3);
}

TEST_F(UMACoreDataFlowTestFixture, uma_core_dataflow_test3) {
	testUmaCoreDataFlow(2, 4);
}

TEST_F(UMACoreDataFlowTestFixture, uma_core_dataflow_test4) {
	testUmaCoreDataFlow(0, 4);
}

TEST_F(SensorValuePointerConvertionTestFixture, value_pointer_convertion) {
	testPointerToNull();
	testPointersToValues();
	testValuesToPointers();
}

TEST(sensor_test, set_amper_list) {
	pair<string, string> p1 = { "s1", "cs1" };
	pair<string, string> p2 = { "s2", "cs2" };
	Sensor *s1 = new Sensor(p1, nullptr, 1.0, 0);
	Sensor *s2 = new Sensor(p2, nullptr, 1.0, 1);
	s1->setAmperList(0);
	s1->setAmperList(1);
	s2->setAmperList(2);
	s2->setAmperList(3);
	s2->setAmperList(4);
	vector<int> v1 = { 0, 1 };
	vector<int> v2 = { 2, 3, 4 };
	vector<int> v3 = { 0, 1, 2, 3, 4 };
	EXPECT_EQ(s1->getAmperList(), v1);
	EXPECT_EQ(s2->getAmperList(), v2);
	s1->setAmperList(s2);
	EXPECT_EQ(s1->getAmperList(), v3);

	delete s1;
	delete s2;
}

TEST(sensor_test, generateDelayedSignal) {
	Agent *agent = new Agent("agent", nullptr);
	Snapshot *snapshot = agent->createSnapshot("snapshot");
	vector<vector<double>> w;
	vector<vector<bool>> b;
	std::pair<string, string> p0 = { "s0", "cs0" };
	std::pair<string, string> p1 = { "s1", "cs1" };
	std::pair<string, string> p2 = { "s2", "cs2" };
	std::pair<string, string> p3 = { "s3", "cs3" };
	vector<std::pair<string, string>> p4 = { { "s4", "cs4" } };
	vector<std::pair<string, string>> p5 = { { "s5", "cs5" } };
	vector<double> diag;
	snapshot->createSensor(p0, diag, w, b);
	snapshot->createSensor(p1, diag, w, b);
	snapshot->createSensor(p2, diag, w, b);
	snapshot->createSensor(p3, diag, w, b);

	vector<vector<bool>> list1 = { { 0, 0, 0, 1, 0, 1, 0, 1 } };
	vector<vector<bool>> list2 = { { 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 } };
	snapshot->ampers(list1, p4);
	snapshot->delays(list2, p5);

	vector<bool> observe1 = { true, false, true, false, false, true, false, true, true, false, true, false };
	vector<bool> observe2 = { true, false, false, true, false, true, false, true, false, true, true, false };

	Sensor *sensor4 = snapshot->getSensor("s4");
	Sensor *sensor5 = snapshot->getSensor("s5");

	snapshot->getDM()->setObserve(observe1);
	snapshot->getDM()->setObserve(observe1);
	EXPECT_EQ(sensor4->generateDelayedSignal(), false);
	EXPECT_EQ(sensor5->generateDelayedSignal(), true);

	snapshot->getDM()->setObserve(observe2);
	snapshot->getDM()->setObserve(observe2);
	EXPECT_EQ(sensor4->generateDelayedSignal(), true);
	EXPECT_EQ(sensor5->generateDelayedSignal(), false);

	delete agent;
}

TEST(sensor_test, get_set_idx) {
	pair<string, string> p1 = { "s1", "cs1" };
	Sensor *s1 = new Sensor(p1, nullptr, 1.0, 0);
	EXPECT_EQ(s1->getIdx(), 0);
	s1->setIdx(2);
	EXPECT_EQ(s1->getIdx(), 2);

	delete s1;
}

TEST(sensor_test, copy_amper_list_test) {
	pair<string, string> p1 = { "s1", "cs1" };
	pair<string, string> p2 = { "s2", "cs2" };
	pair<string, string> p3 = { "s3", "cs3" };
	pair<string, string> p4 = { "s4", "cs4" };
	pair<string, string> p5 = { "s5", "cs5" };

	Sensor *s1 = new Sensor(p1, nullptr, 1.0, 0);
	Sensor *s2 = new Sensor(p2, nullptr, 1.0, 1);
	Sensor *s3 = new Sensor(p3, nullptr, 1.0, 2);
	Sensor *s4 = new Sensor(p4, nullptr, 1.0, 3);
	Sensor *s5 = new Sensor(p5, nullptr, 1.0, 4);

	bool *ampers = new bool[30];
	ampers[0] = 0; ampers[1] = 0;
	ampers[2] = 0; ampers[3] = 0; ampers[4] = 0; ampers[5] = 0;
	ampers[6] = 0; ampers[7] = 0; ampers[8] = 0; ampers[9] = 0; ampers[10] = 0; ampers[11] = 0;
	ampers[12] = 0; ampers[13] = 0; ampers[14] = 0; ampers[15] = 0; ampers[16] = 0; ampers[17] = 0; ampers[18] = 0; ampers[19] = 0;
	ampers[20] = 0; ampers[21] = 0; ampers[22] = 0; ampers[23] = 0; ampers[24] = 0; ampers[25] = 0; ampers[26] = 0; ampers[27] = 0; ampers[28] = 0; ampers[29] = 0;

	s4->setAmperList(0);
	s4->setAmperList(3);
	s4->setAmperList(5);
	s4->setAmperList(6);

	s4->copyAmperList(ampers);

	EXPECT_EQ(ampers[12], 1);
	EXPECT_EQ(ampers[13], 0);
	EXPECT_EQ(ampers[14], 0);
	EXPECT_EQ(ampers[15], 1);
	EXPECT_EQ(ampers[16], 0);
	EXPECT_EQ(ampers[17], 1);
	EXPECT_EQ(ampers[18], 1);
	EXPECT_EQ(ampers[19], 0);

	delete s1;
	delete s2;
	delete s3;
	delete s4;
	delete s5;
}

TEST_F(SensorSavingLoading, sensor_save_load) {
	savingAndLoading();
	copying();
}

TEST(sensor_pair_test, sensor_pair_test) {
	pair<string, string> p1 = { "s1", "cs1" };

	Sensor *s1 = new Sensor(p1, nullptr, 1.0, 0);

	double *thresholds = new double;
	double *weights = new double[4];
	bool *dirs = new bool[4];

	SensorPair *sp = new SensorPair(nullptr, s1, s1, 0.25, 1.0);

	sp->setAllPointers(weights, dirs, thresholds);
	sp->valuesToPointers();

	EXPECT_DOUBLE_EQ(sp->getThreshold(), 0.25);
	sp->setThreshold(0.5);
	EXPECT_DOUBLE_EQ(sp->getThreshold(), 0.5);

	AttrSensorPair *asp1 = sp->getAttrSensorPair(true, true);
	AttrSensorPair *asp2 = sp->getAttrSensorPair(true, false);
	EXPECT_DOUBLE_EQ(asp1->getW(), 0.25);
	EXPECT_DOUBLE_EQ(asp2->getW(), 0.25);
	EXPECT_DOUBLE_EQ(asp1->getD(), true);
	EXPECT_DOUBLE_EQ(asp2->getD(), false);
	asp1->setD(false);
	asp1->setW(0.8);
	EXPECT_DOUBLE_EQ(asp1->getW(), 0.8);
	EXPECT_DOUBLE_EQ(asp1->getD(), false);

	delete s1;
	delete sp;
	delete thresholds;
	delete dirs;
	delete weights;
}

TEST_F(SensorPairSavingLoading, sensor_pair_save_load) {
	savingAndLoading();
	copying();
}

TEST(attr_sensor_test, attr_sensor_test) {
	AttrSensor *as = new AttrSensor("attr_sensor", nullptr, 0, true, 0.5);
	double *diag = new double[2];
	double *diag_ = new double[2];
	bool *observe = new bool[2];
	bool *observe_ = new bool[2];
	bool *target = new bool[2];

	EXPECT_EQ(as->getIdx(), 0);
	as->setIdx(1);
	EXPECT_EQ(as->getIdx(), 1);

	EXPECT_EQ(as->getIsOriginPure(), true);
	as->setIsOriginPure(false);
	EXPECT_EQ(as->getIsOriginPure(), false);

	as->setDiagPointers(diag, diag_);
	as->setObservePointers(observe, observe_);
	as->setTargetPointers(target);
	as->valuesToPointers();

	EXPECT_EQ(as->getDiag(), 0.5);
	EXPECT_EQ(as->getOldDiag(), 0.5);
	EXPECT_EQ(as->getTarget(), false);

	as->setDiag(0.6);
	as->setOldDiag(0.7);
	EXPECT_EQ(as->getDiag(), 0.6);
	EXPECT_EQ(as->getOldDiag(), 0.7);

	delete as;
	delete[] diag, diag_;
	delete[] observe, observe_;
}

TEST_F(AttrSensorSavingLoading, attr_sensor_save_load) {
	savingAndLoading();
	copying();
}

TEST_F(AttrSensorPairSavingLoading, attr_sensor_pair_save_load) {
	savingAndLoading();
	copying();
}

TEST(UMACoreObject_layering_test, UMACoreObject_layering_test) {
	PropertyPage *pp = CoreService::instance()->getPropertyPage();

	PropertyMap *worldMap = pp->get("World");
	if (!worldMap) {
		pp->add("World", new PropertyMap());
		worldMap = pp->get("World");
	}

	PropertyMap *experimentMap = pp->get("Experiment");
	if (!experimentMap) {
		pp->add("Experiment", new PropertyMap());
		experimentMap = pp->get("Experiment");
	}

	PropertyMap *agentMap = pp->get("Agent");
	if (!agentMap) {
		pp->add("Agent", new PropertyMap());
		agentMap = pp->get("Agent");
	}

	PropertyMap *agentQualitativeMap = pp->get("Agent::Qualitative");
	if (!agentQualitativeMap) {
		pp->add("Agent::Qualitative", new PropertyMap());
		agentQualitativeMap = pp->get("Agent::Qualitative");
	}

	PropertyMap *snapshotMap = pp->get("Snapshot");
	if (!snapshotMap) {
		pp->add("Snapshot", new PropertyMap());
		snapshotMap = pp->get("Snapshot");
	}

	PropertyMap *snapshotQualitativeMap = pp->get("Snapshot::Qualitative");
	if (!snapshotQualitativeMap) {
		pp->add("Snapshot::Qualitative", new PropertyMap());
		snapshotQualitativeMap = pp->get("Snapshot::Qualitative");
	}
	
	worldMap->add("a", "1");
	worldMap->add("b", "-1");
	worldMap->add("c", "-1");
	worldMap->add("d", "-1");
	experimentMap->add("b", "2");
	experimentMap->add("c", "-1");
	experimentMap->add("d", "-1");
	experimentMap->add("e", "5");
	agentMap->add("c", "3");
	agentMap->add("d", "-1");
	agentMap->add("f", "6");
	agentQualitativeMap->add("d", "10");
	agentQualitativeMap->add("g", "7");
	snapshotMap->add("d", "4");
	snapshotMap->add("h", "8");
	snapshotQualitativeMap->add("h", "11");
	snapshotQualitativeMap->add("i", "9");
	
	World::reset();
	Experiment *experiment = World::instance()->createExperiment("test_experiment");
	Agent *agent = experiment->createAgent("test_agent", UMACoreConstant::UMA_AGENT::AGENT_STATIONARY);
	Agent *agentQualitative = experiment->createAgent("test_agent_qualitative", UMACoreConstant::UMA_AGENT::AGENT_QUALITATIVE);
	Snapshot *snapshot = agent->createSnapshot("test_snapshot");
	Snapshot *snapshotQualitative = agentQualitative->createSnapshot("test_snapshot_qualitative");
	
	EXPECT_EQ(agent->getPropertyMap()->get("a"), "1");
	EXPECT_EQ(agent->getPropertyMap()->get("b"), "2");
	EXPECT_EQ(agent->getPropertyMap()->get("c"), "3");
	EXPECT_EQ(agent->getPropertyMap()->get("d"), "-1");
	EXPECT_EQ(agent->getPropertyMap()->get("e"), "5");
	EXPECT_EQ(agent->getPropertyMap()->get("f"), "6");

	EXPECT_EQ(agentQualitative->getPropertyMap()->get("a"), "1");
	EXPECT_EQ(agentQualitative->getPropertyMap()->get("b"), "2");
	EXPECT_EQ(agentQualitative->getPropertyMap()->get("c"), "3");
	EXPECT_EQ(agentQualitative->getPropertyMap()->get("d"), "10");
	EXPECT_EQ(agentQualitative->getPropertyMap()->get("e"), "5");
	EXPECT_EQ(agentQualitative->getPropertyMap()->get("f"), "6");
	EXPECT_EQ(agentQualitative->getPropertyMap()->get("g"), "7");

	CoreService::reset();
	World::reset();
	
}

TEST_F(UMASavingLoading, uma_save_load) {
	savingAndLoading();
}

TEST_F(UMAAgentCopying, uma_agent_copying) {
	copyingAgents();
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	(void)RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}