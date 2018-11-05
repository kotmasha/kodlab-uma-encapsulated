#include "UMACoreTestFixture.h"
#include "PropertyMap.h"
#include "UMAutil.h"

extern int ind(int row, int col);
extern int compi(int x);

WorldTestFixture::WorldTestFixture() {}

WorldTestFixture::~WorldTestFixture() {}

SnapshotUpdateQTestFixture::SnapshotUpdateQTestFixture() {
	agentStationary = new Agent("agentStationary", nullptr);
	snapshotStationary = agentStationary->createSnapshot("snapshotStationary");
	snapshotStationary->_ppm->add("q", "1");

	agentQualitative = new AgentQualitative("agentQualitative", nullptr);
	snapshotQualitative = agentQualitative->createSnapshot("snapshotQualitative");
	snapshotQualitative->_ppm->add("q", "2");

	agentDiscounted = new AgentDiscounted("agentDiscounted", nullptr);
	snapshotDiscounted = agentDiscounted->createSnapshot("snapshotDiscounted");
	snapshotDiscounted->_ppm->add("q", "3");

	agentEmpirical = new AgentEmpirical("agentEmpirical", nullptr);
	snapshotEmpirical = agentEmpirical->createSnapshot("snapshotEmpirical");
}

SnapshotUpdateQTestFixture::~SnapshotUpdateQTestFixture() {
	delete agentStationary, agentQualitative, agentDiscounted, agentEmpirical;
}

void SnapshotUpdateQTestFixture::testUpdateQ() {
	EXPECT_EQ(0, snapshotStationary->getQ());// value of q is hard-coded to be 0
	snapshotStationary->updateQ();
	EXPECT_EQ(1, snapshotStationary->getQ());

	EXPECT_EQ(0, snapshotQualitative->getQ());
	snapshotQualitative->updateQ();
	EXPECT_EQ(0, snapshotQualitative->getQ());

	EXPECT_EQ(0, snapshotDiscounted->getQ());
	snapshotDiscounted->updateQ();
	EXPECT_EQ(3, snapshotDiscounted->getQ());

	EXPECT_EQ(0, snapshotEmpirical->getQ());
	snapshotEmpirical->setQ(1.5);
	snapshotEmpirical->updateQ();
	EXPECT_EQ(2, snapshotEmpirical->getQ());
}

AmperAndTestFixture::AmperAndTestFixture(){
	agent = new Agent("agent", nullptr);
	snapshot = agent->createSnapshot("snapshot");
	snapshot->setTotal(1);
	snapshot->setOldTotal(1);
	vector<vector<double>> w0 = { { 0.2, 0.0, 0.0, 0.8 } };
	vector<vector<double>> w1 = { { 0.2, 0.2, 0.0, 0.6 },{ 0.4, 0.0, 0.0, 0.6 } };
	vector<vector<double>> w2 = { { 0.2, 0.4, 0.0, 0.4 },{ 0.4, 0.2, 0.0, 0.4 },{ 0.6, 0.0, 0.0, 0.4 } };
	vector<vector<double>> w3 = { { 0.2, 0.6, 0.0, 0.2 },{ 0.4, 0.4, 0.0, 0.2 },{ 0.6, 0.2, 0.0, 0.2 },{ 0.8, 0.0, 0.0, 0.2 } };
	vector<vector<bool>> b0 = { { true, false, false, true } };
	vector<vector<bool>> b1 = { { false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b2 = { { false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b3 = { { false, false, false, false },{ false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	std::pair<string, string> p0 = { "s0", "cs0" };
	std::pair<string, string> p1 = { "s1", "cs1" };
	std::pair<string, string> p2 = { "s2", "cs2" };
	std::pair<string, string> p3 = { "s3", "cs3" };
	vector<double> diag;
	snapshot->createSensor(p0, diag, w0, b0);
	snapshot->createSensor(p1, diag, w1, b1);
	snapshot->createSensor(p2, diag, w2, b2);
	snapshot->createSensor(p3, diag, w3, b3);
}

AmperAndTestFixture::~AmperAndTestFixture() {
	delete agent;
}

vector<vector<double>> AmperAndTestFixture::testAmperAnd(int mid1, int mid2, bool merge, std::pair<string, string> &p) {
	snapshot->ampersand(mid1, mid2, merge, p);

	vector<vector<double>> w;
	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			AttrSensorPair *asp = snapshot->getAttrSensorPair(i, j);
			tmp.push_back(asp->_vw);
		}
		w.push_back(tmp);
	}
	
	return w;
}

GenerateDelayedWeightsTestFixture::GenerateDelayedWeightsTestFixture() {
	agent = new Agent("agent", nullptr);
	snapshot = agent->createSnapshot("snapshot");
	agentQualitative = new AgentQualitative("agentQualitative", nullptr);
	snapshotQualitative = agentQualitative->createSnapshot("snapshotQualitative");
	vector<vector<double>> w0 = { { 0.2, 0.0, 0.0, 0.8 } };
	vector<vector<double>> w1 = { { 0.2, 0.2, 0.0, 0.6 },{ 0.4, 0.0, 0.0, 0.6 } };
	vector<vector<double>> w2 = { { 0.2, 0.4, 0.0, 0.4 },{ 0.4, 0.2, 0.0, 0.4 },{ 0.6, 0.0, 0.0, 0.4 } };
	vector<vector<double>> w3 = { { 0.2, 0.6, 0.0, 0.2 },{ 0.4, 0.4, 0.0, 0.2 },{ 0.6, 0.2, 0.0, 0.2 },{ 0.8, 0.0, 0.0, 0.2 } };
	vector<vector<bool>> b0 = { { true, false, false, true } };
	vector<vector<bool>> b1 = { { false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b2 = { { false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b3 = { { false, false, false, false },{ false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	std::pair<string, string> p0 = { "s0", "cs0" };
	std::pair<string, string> p1 = { "s1", "cs1" };
	std::pair<string, string> p2 = { "s2", "cs2" };
	std::pair<string, string> p3 = { "s3", "cs3" };
	vector<double> diag;
	Sensor *s0 = snapshot->createSensor(p0, diag, w0, b0);
	Sensor *s1 = snapshot->createSensor(p1, diag, w1, b1);
	Sensor *s2 = snapshot->createSensor(p2, diag, w2, b2);
	Sensor *s3 = snapshot->createSensor(p3, diag, w3, b3);
	
	snapshot->getAttrSensor(0)->_vdiag = 0.2;
	snapshot->getAttrSensor(1)->_vdiag = 0.8;
	snapshot->getAttrSensor(2)->_vdiag = 0.4;
	snapshot->getAttrSensor(3)->_vdiag = 0.6;
	snapshot->getAttrSensor(4)->_vdiag = 0.6;
	snapshot->getAttrSensor(5)->_vdiag = 0.4;
	snapshot->getAttrSensor(6)->_vdiag = 0.8;
	snapshot->getAttrSensor(7)->_vdiag = 0.2;

	snapshotQualitative->createSensor(p0, diag, w0, b0);
	snapshotQualitative->createSensor(p1, diag, w1, b1);
	snapshotQualitative->createSensor(p2, diag, w2, b2);
	snapshotQualitative->createSensor(p3, diag, w3, b3);

	snapshotQualitative->getAttrSensor(0)->_vdiag = -1;
	snapshotQualitative->getAttrSensor(1)->_vdiag = -1;
	snapshotQualitative->getAttrSensor(2)->_vdiag = -1;
	snapshotQualitative->getAttrSensor(3)->_vdiag = -1;
	snapshotQualitative->getAttrSensor(4)->_vdiag = -1;
	snapshotQualitative->getAttrSensor(5)->_vdiag = -1;
	snapshotQualitative->getAttrSensor(6)->_vdiag = -1;
	snapshotQualitative->getAttrSensor(7)->_vdiag = -1;
}

GenerateDelayedWeightsTestFixture::~GenerateDelayedWeightsTestFixture() {
	delete agent, agentQualitative;
}

vector<vector<double>> GenerateDelayedWeightsTestFixture::testGenerateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair
	, vector<bool> &observe, UMA_AGENT type) {
	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		*(snapshot->getAttrSensor(i)->_observe) = observe[i];
		*(snapshot->getAttrSensor(i)->_observe_) = observe[i];
	}
	Snapshot *currentSnapshot = nullptr;
	if (UMA_AGENT::AGENT_QUALITATIVE == type) {
		currentSnapshot = snapshotQualitative;
	}
	else {
		currentSnapshot = snapshot;
	}

	currentSnapshot->generateDelayedWeights(mid, merge, idPair);

	vector<vector<double>> w;
	for (int i = 0; i < 2 * currentSnapshot->_sensors.size(); ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			AttrSensorPair *asp = currentSnapshot->getAttrSensorPair(i, j);
			tmp.push_back(asp->_vw);
		}
		w.push_back(tmp);
	}

	return w;
}

AmperTestFixture::AmperTestFixture() {
	agent = new Agent("agent", nullptr);
	snapshot = agent->createSnapshot("snapshot");
	snapshot->setTotal(1);
	snapshot->setOldTotal(1);

	vector<vector<double>> w0 = { { 0.2, 0.0, 0.0, 0.8 } };
	vector<vector<double>> w1 = { { 0.2, 0.2, 0.0, 0.6 },{ 0.4, 0.0, 0.0, 0.6 } };
	vector<vector<double>> w2 = { { 0.2, 0.4, 0.0, 0.4 },{ 0.4, 0.2, 0.0, 0.4 },{ 0.6, 0.0, 0.0, 0.4 } };
	vector<vector<double>> w3 = { { 0.2, 0.6, 0.0, 0.2 },{ 0.4, 0.4, 0.0, 0.2 },{ 0.6, 0.2, 0.0, 0.2 },{ 0.8, 0.0, 0.0, 0.2 } };
	vector<vector<bool>> b0 = { { true, false, false, true } };
	vector<vector<bool>> b1 = { { false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b2 = { { false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b3 = { { false, false, false, false },{ false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	std::pair<string, string> p0 = { "s0", "cs0" };
	std::pair<string, string> p1 = { "s1", "cs1" };
	std::pair<string, string> p2 = { "s2", "cs2" };
	std::pair<string, string> p3 = { "s3", "cs3" };
	vector<double> diag;
	snapshot->createSensor(p0, diag, w0, b0);
	snapshot->createSensor(p1, diag, w1, b1);
	snapshot->createSensor(p2, diag, w2, b2);
	snapshot->createSensor(p3, diag, w3, b3);
}

AmperTestFixture::~AmperTestFixture() {
	delete agent;
}

vector<vector<double>> AmperTestFixture::testAmper(const vector<int> &list, const std::pair<string, string> &uuid) {
	snapshot->amper(list, uuid);

	vector<vector<double>> w;
	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			AttrSensorPair *asp = snapshot->getAttrSensorPair(i, j);
			tmp.push_back(asp->_vw);
		}
		w.push_back(tmp);
	}

	return w;
}

UMACoreDataFlowTestFixture::UMACoreDataFlowTestFixture() {
	dm = new DataManager(nullptr);

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> d1 = { 0.2, 0.8 };
	vector<vector<double>> w1, w3;
	vector<vector<double>> w0 = { { 0.2, 0.8, 0.457, 0.543 } };
	vector<vector<double>> w2 = { { 0.2, 0.8, 0.4, 0.6 },{ 0.1, 0.9, 0, 1 },{ 0.3, 0.7, 0.8, 0.2 } };
	vector<vector<bool>> b1, b3;
	vector<vector<bool>> b0 = { { true, false, true, false } };
	vector<vector<bool>> b2 = { { true, false, false, true },{ false, false, false, false },{ true, true, true, true } };

	weights = { 0.2, 0.457, 0.543, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.8, 0.1, 0.9, 0.3, 0.4, 0.6,
	0.0, 1.0, 0.8, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 };
	thresholds = { 0.125, 0.25, 0.25, 0.125, 0.125, 0.125, 0.5, 0.5, 0.5, 0.5 };
	observe = { true, false, false, true, false, true, true, false };

	dirs = { 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1 };
	diag = { 0.5, 0.5, 0.2, 0.8, 0.5, 0.5, 0.5, 0.5 };

	sensors.push_back(new Sensor(sensor1, nullptr, 1.0, 0));
	sensors.push_back(new Sensor(sensor2, nullptr, d1, 1));
	sensors.push_back(new Sensor(sensor3, nullptr, 1.0, 2));
	sensors.push_back(new Sensor(sensor4, nullptr, 1.0, 3));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[0], sensors[0], 0.125, w0[0], b0[0]));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[1], sensors[0], 0.25, 1.0));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[1], sensors[1], 0.25, 1.0));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[2], sensors[0], 0.125, w2[0], b2[0]));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[2], sensors[1], 0.125, w2[1], b2[1]));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[2], sensors[2], 0.125, w2[2], b2[2]));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[3], sensors[0], 0.5, 1.0));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[3], sensors[1], 0.5, 1.0));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[3], sensors[2], 0.5, 1.0));
	sensorPairs.push_back(new SensorPair(nullptr, sensors[3], sensors[3], 0.5, 1.0));

	sensors[0]->_m->_vobserve = true; sensors[0]->_m->_vobserve_ = true;
	sensors[0]->_cm->_vobserve = false; sensors[0]->_cm->_vobserve_ = false;
	sensors[1]->_m->_vobserve = false; sensors[1]->_m->_vobserve_ = false;
	sensors[1]->_cm->_vobserve = true; sensors[1]->_cm->_vobserve_ = true;
	sensors[2]->_m->_vobserve = false; sensors[2]->_m->_vobserve_ = false;
	sensors[2]->_cm->_vobserve = true; sensors[2]->_cm->_vobserve_ = true;
	sensors[3]->_m->_vobserve = true; sensors[3]->_m->_vobserve_ = true;
	sensors[3]->_cm->_vobserve = false; sensors[3]->_cm->_vobserve_ = false;

	double total = 1.0;
	dm->reallocateMemory(total, 4);
}

UMACoreDataFlowTestFixture::~UMACoreDataFlowTestFixture(){
	delete dm;
	for (int i = 0; i < sensors.size(); ++i) {
		delete sensors[i];
		sensors[i] = NULL;
	}
	for (int i = 0; i < sensorPairs.size(); ++i) {
		delete sensorPairs[i];
		sensorPairs[i] = NULL;
	}
}

void UMACoreDataFlowTestFixture::testUmaCoreDataFlow(int startIdx, int endIdx) {
	dm->createSensorsToArraysIndex(startIdx, endIdx, sensors);
	dm->createSensorPairsToArraysIndex(startIdx, endIdx, sensorPairs);

	for (int i = startIdx; i < endIdx; ++i) {
		EXPECT_EQ(sensors[i]->_m->_diag, dm->h_diag + 2 * i);
		EXPECT_EQ(sensors[i]->_cm->_diag, dm->h_diag + 2 * i + 1);
		EXPECT_EQ(sensors[i]->_m->_diag_, dm->h_diag_ + 2 * i);
		EXPECT_EQ(sensors[i]->_cm->_diag_, dm->h_diag_ + 2 * i + 1);
		EXPECT_EQ(sensors[i]->_m->_observe, dm->h_observe + 2 * i);
		EXPECT_EQ(sensors[i]->_cm->_observe, dm->h_observe + 2 * i + 1);
		EXPECT_EQ(sensors[i]->_m->_observe_, dm->h_observe_ + 2 * i);
		EXPECT_EQ(sensors[i]->_cm->_observe_, dm->h_observe_ + 2 * i + 1);
		EXPECT_EQ(sensors[i]->_m->_current, dm->h_current + 2 * i);
		EXPECT_EQ(sensors[i]->_cm->_current, dm->h_current + 2 * i + 1);
		EXPECT_EQ(sensors[i]->_m->_target, dm->h_target + 2 * i);
		EXPECT_EQ(sensors[i]->_cm->_target, dm->h_target + 2 * i + 1);
		EXPECT_EQ(sensors[i]->_m->_prediction, dm->h_prediction + 2 * i);
		EXPECT_EQ(sensors[i]->_cm->_prediction, dm->h_prediction + 2 * i + 1);
	}
	for (int i = ind(startIdx, 0); i < ind(endIdx, 0); ++i) {
		int idxI = 2 * sensorPairs[i]->_sensor_i->_idx;
		int idxJ = 2 * sensorPairs[i]->_sensor_j->_idx;
		EXPECT_EQ(sensorPairs[i]->mij->_w, dm->h_weights + ind(idxI, idxJ));
		EXPECT_EQ(sensorPairs[i]->mi_j->_w, dm->h_weights + ind(idxI, idxJ + 1));
		EXPECT_EQ(sensorPairs[i]->m_ij->_w, dm->h_weights + ind(idxI + 1, idxJ));
		EXPECT_EQ(sensorPairs[i]->m_i_j->_w, dm->h_weights + ind(idxI + 1, idxJ + 1));

		EXPECT_EQ(sensorPairs[i]->mij->_d, dm->h_dirs + ind(idxI, idxJ));
		EXPECT_EQ(sensorPairs[i]->mi_j->_d, dm->h_dirs + ind(idxI, idxJ + 1));
		EXPECT_EQ(sensorPairs[i]->m_ij->_d, dm->h_dirs + ind(idxI + 1, idxJ));
		EXPECT_EQ(sensorPairs[i]->m_i_j->_d, dm->h_dirs + ind(idxI + 1, idxJ + 1));

		EXPECT_EQ(sensorPairs[i]->_threshold, dm->h_thresholds + ind(idxI / 2, idxJ / 2));
	}

	dm->copySensorsToArrays(startIdx, endIdx, sensors);
	dm->copySensorPairsToArrays(startIdx, endIdx, sensorPairs);
	dm->copyArraysToSensors(startIdx, endIdx, sensors);
	dm->copyArraysToSensorPairs(startIdx, endIdx, sensorPairs);

	for (int i = startIdx; i < endIdx; ++i) {
		EXPECT_EQ(diag[2 * i], sensors[i]->_m->_vdiag);
		EXPECT_EQ(diag[2 * i + 1], sensors[i]->_cm->_vdiag);
		EXPECT_EQ(diag[2 * i], sensors[i]->_m->_vdiag_);
		EXPECT_EQ(diag[2 * i + 1], sensors[i]->_cm->_vdiag_);
		EXPECT_EQ(observe[2 * i], sensors[i]->_m->_vobserve);
		EXPECT_EQ(observe[2 * i], sensors[i]->_m->_vobserve_);
		EXPECT_EQ(observe[2 * i + 1], sensors[i]->_cm->_vobserve);
		EXPECT_EQ(observe[2 * i + 1], sensors[i]->_cm->_vobserve_);
	}

	for (int i = ind(startIdx, 0); i < ind(endIdx, 0); ++i) {
		int idxI = 2 * sensorPairs[i]->_sensor_i->_idx;
		int idxJ = 2 * sensorPairs[i]->_sensor_j->_idx;
		EXPECT_DOUBLE_EQ(weights[ind(idxI, idxJ)], sensorPairs[i]->mij->_vw);
		EXPECT_DOUBLE_EQ(weights[ind(idxI, idxJ + 1)], sensorPairs[i]->mi_j->_vw);
		EXPECT_DOUBLE_EQ(weights[ind(idxI + 1, idxJ)], sensorPairs[i]->m_ij->_vw);
		EXPECT_DOUBLE_EQ(weights[ind(idxI + 1, idxJ + 1)], sensorPairs[i]->m_i_j->_vw);

		EXPECT_EQ(dirs[ind(idxI, idxJ)], sensorPairs[i]->mij->_vd);
		EXPECT_EQ(dirs[ind(idxI, idxJ + 1)], sensorPairs[i]->mi_j->_vd);
		EXPECT_EQ(dirs[ind(idxI + 1, idxJ)], sensorPairs[i]->m_ij->_vd);
		EXPECT_EQ(dirs[ind(idxI + 1, idxJ + 1)], sensorPairs[i]->m_i_j->_vd);

		EXPECT_DOUBLE_EQ(thresholds[ind(idxI / 2, idxJ / 2)], sensorPairs[i]->_vthreshold);
	}
}

SensorValuePointerConvertionTestFixture::SensorValuePointerConvertionTestFixture() {
	pair<string, string> p1 = { "s1", "cs1" };
	pair<string, string> p2 = { "s2", "cs2" };
	pair<string, string> p3 = { "s3", "cs3" };
	pair<string, string> p4 = { "s4", "cs4" };
	s1 = new Sensor(p1, nullptr, 1.0, 0);
	s2 = new Sensor(p2, nullptr, 1.0, 1);
	s3 = new Sensor(p3, nullptr, 1.0, 2);
	s4 = new Sensor(p4, nullptr, 1.0, 3);

	s = { s1, s2, s3, s4 };

	h_diag = new double[8];
	h_diag_ = new double[8];
	h_observe = new bool[8];
	h_observe_ = new bool[8];
	h_current = new bool[8];
	h_current_ = new bool[8];
	h_target = new bool[8];
	h_prediction = new bool[8];

	for (int i = 0; i < 8; ++i) {
		h_diag[i] = i * 0.1;
		h_diag_[i] = i * 0.2;
		h_observe[i] = i % 2;
		h_observe_[i] = i % 2;
		h_current[i] = i % 2;
		h_current_[i] = i % 2;
		h_target[i] = !i % 2;
		h_prediction[i] = !i % 2;
	}
}

SensorValuePointerConvertionTestFixture::~SensorValuePointerConvertionTestFixture() {
	delete h_diag, h_diag_, h_observe, h_observe_, h_current, h_current_, h_target, h_prediction;
	delete s1, s2, s3, s4;
}

void SensorValuePointerConvertionTestFixture::testPointerToNull() {
	for (int i = 0; i < s.size(); ++i) {
		s[i]->pointersToNull();

		EXPECT_EQ(nullptr, s[i]->_m->_diag);
		EXPECT_EQ(nullptr, s[i]->_cm->_diag);
		EXPECT_EQ(nullptr, s[i]->_m->_diag_);
		EXPECT_EQ(nullptr, s[i]->_cm->_diag_);
		EXPECT_EQ(nullptr, s[i]->_m->_observe);
		EXPECT_EQ(nullptr, s[i]->_cm->_observe);
		EXPECT_EQ(nullptr, s[i]->_m->_observe_);
		EXPECT_EQ(nullptr, s[i]->_cm->_observe_);
		EXPECT_EQ(nullptr, s[i]->_m->_current);
		EXPECT_EQ(nullptr, s[i]->_cm->_current);
		EXPECT_EQ(nullptr, s[i]->_m->_target);
		EXPECT_EQ(nullptr, s[i]->_cm->_target);
		EXPECT_EQ(nullptr, s[i]->_m->_prediction);
		EXPECT_EQ(nullptr, s[i]->_cm->_prediction);
	}
}

void SensorValuePointerConvertionTestFixture::testPointersToValues() {
	for (int i = 0; i < s.size(); ++i) {
		s[i]->setAttrSensorDiagPointers(h_diag, h_diag_);
		s[i]->setAttrSensorObservePointers(h_observe, h_observe_);
		s[i]->setAttrSensorCurrentPointers(h_current, h_current_);
		s[i]->setAttrSensorTargetPointers(h_target);
		s[i]->setAttrSensorPredictionPointers(h_prediction);

		EXPECT_EQ(h_diag + 2 * i, s[i]->_m->_diag);
		EXPECT_EQ(h_diag + 2 * i + 1, s[i]->_cm->_diag);
		EXPECT_EQ(h_diag_ + 2 * i, s[i]->_m->_diag_);
		EXPECT_EQ(h_diag_ + 2 * i + 1, s[i]->_cm->_diag_);
		EXPECT_EQ(h_observe + 2 * i, s[i]->_m->_observe);
		EXPECT_EQ(h_observe + 2 * i + 1, s[i]->_cm->_observe);
		EXPECT_EQ(h_observe_ + 2 * i, s[i]->_m->_observe_);
		EXPECT_EQ(h_observe_ + 2 * i + 1, s[i]->_cm->_observe_);
		EXPECT_EQ(h_current + 2 * i, s[i]->_m->_current);
		EXPECT_EQ(h_current + 2 * i + 1, s[i]->_cm->_current);
		EXPECT_EQ(h_target + 2 * i, s[i]->_m->_target);
		EXPECT_EQ(h_target + 2 * i + 1, s[i]->_cm->_target);
		EXPECT_EQ(h_prediction + 2 * i, s[i]->_m->_prediction);
		EXPECT_EQ(h_prediction + 2 * i + 1, s[i]->_cm->_prediction);

		s[i]->pointersToValues();

		EXPECT_EQ(h_diag[2 * i], s[i]->_m->_vdiag);
		EXPECT_EQ(h_diag[2 * i + 1], s[i]->_cm->_vdiag);
		EXPECT_EQ(h_diag_[2 * i], s[i]->_m->_vdiag_);
		EXPECT_EQ(h_diag_[2 * i + 1], s[i]->_cm->_vdiag_);
		EXPECT_EQ(h_observe[2 * i], s[i]->_m->_vobserve);
		EXPECT_EQ(h_observe[2 * i + 1], s[i]->_cm->_vobserve);
		EXPECT_EQ(h_observe_[2 * i], s[i]->_m->_vobserve_);
		EXPECT_EQ(h_observe_[2 * i + 1], s[i]->_cm->_vobserve_);
		EXPECT_EQ(h_target[2 * i], s[i]->_m->_vtarget);
		EXPECT_EQ(h_target[2 * i + 1], s[i]->_cm->_vtarget);
	}
}

void SensorValuePointerConvertionTestFixture::testValuesToPointers() {
	for (int i = 0; i < s.size(); ++i) {
		s[i]->_m->_vdiag = 0.1;
		s[i]->_cm->_vdiag = 0.2;
		s[i]->_m->_vdiag_ = 0.3;
		s[i]->_cm->_vdiag_ = 0.4;
		s[i]->_m->_vobserve = true;
		s[i]->_cm->_vobserve = false;
		s[i]->_m->_vobserve_ = false;
		s[i]->_cm->_vobserve_ = true;
		s[i]->_m->_vtarget = true;
		s[i]->_cm->_vtarget = false;

		s[i]->valuesToPointers();

		EXPECT_EQ(h_diag[2 * i], s[i]->_m->_vdiag);
		EXPECT_EQ(h_diag[2 * i + 1], s[i]->_cm->_vdiag);
		EXPECT_EQ(h_diag_[2 * i], s[i]->_m->_vdiag_);
		EXPECT_EQ(h_diag_[2 * i + 1], s[i]->_cm->_vdiag_);
		EXPECT_EQ(h_observe[2 * i], s[i]->_m->_vobserve);
		EXPECT_EQ(h_observe[2 * i + 1], s[i]->_cm->_vobserve);
		EXPECT_EQ(h_observe_[2 * i], s[i]->_m->_vobserve_);
		EXPECT_EQ(h_observe_[2 * i + 1], s[i]->_cm->_vobserve_);
		EXPECT_EQ(h_target[2 * i], s[i]->_m->_vtarget);
		EXPECT_EQ(h_target[2 * i + 1], s[i]->_cm->_vtarget);
	}
}

SensorSavingLoading::SensorSavingLoading() {
	fileName = "sensor_test.uma";
	pair<string, string> p1 = { "s", "cs" };
	s1 = new Sensor(p1, nullptr, 1.0, 10);

	s1->_amper.push_back(0);
	s1->_amper.push_back(2);
	s1->_amper.push_back(5);
}

SensorSavingLoading::~SensorSavingLoading() {
	SysUtil::UMARemove(fileName);

	delete s1, s2;
}

void SensorSavingLoading::savingAndLoading() {
	ofstream output;
	output.open(fileName, ios::binary | ios::out);
	s1->saveSensor(output);
	output.close();

	ifstream input;
	input.open(fileName, ios::binary | ios::in);
	s2 = Sensor::loadSensor(input, nullptr);
	input.close();

	EXPECT_EQ(s1->_uuid, s2->_uuid);
	EXPECT_EQ(s1->_idx, s2->_idx);
	EXPECT_EQ(s1->_amper, s2->_amper);
}

AttrSensorSavingLoading::AttrSensorSavingLoading() {
	fileName = "attr_sensor_test.uma";
	const string uuid = "attr_sensor_0";
	as1 = new AttrSensor(uuid, nullptr, 10, true, 10.12);
}

AttrSensorSavingLoading::~AttrSensorSavingLoading() {
	SysUtil::UMARemove(fileName);

	delete as1, as2;
}

void AttrSensorSavingLoading::savingAndLoading() {
	ofstream output;
	output.open(fileName, ios::binary | ios::out);
	as1->saveAttrSensor(output);
	output.close();

	ifstream input;
	input.open(fileName, ios::binary | ios::in);
	as2 = AttrSensor::loadAttrSensor(input, nullptr);
	input.close();

	EXPECT_EQ(as1->_uuid, as2->_uuid);
	EXPECT_EQ(as1->_idx, as2->_idx);
	EXPECT_EQ(as1->_isOriginPure, as2->_isOriginPure);
	EXPECT_EQ(as1->_vdiag, as2->_vdiag);
	EXPECT_EQ(as1->_vdiag_, as2->_vdiag_);
}

SensorPairSavingLoading::SensorPairSavingLoading() {
	fileName = "sensor_pair.uma";
	const string uuid = "attr_sensor_0";

	pair<string, string> p1 = { "s1", "cs1" };
	pair<string, string> p2 = { "s2", "cs2" };
	s1 = new Sensor(p1, nullptr, 1.0, 1);
	s2 = new Sensor(p2, nullptr, 0.5, 3);

	sp1 = new SensorPair(nullptr, s1, s2, 0.123);
}

SensorPairSavingLoading::~SensorPairSavingLoading() {
	SysUtil::UMARemove(fileName);

	delete sp1, sp2;
	delete s1, s2;
}

void SensorPairSavingLoading::savingAndLoading() {
	ofstream output;
	output.open(fileName, ios::binary | ios::out);
	sp1->saveSensorPair(output);
	output.close();

	vector<Sensor*> sensors = {nullptr, s1, nullptr, s2 };

	ifstream input;
	input.open(fileName, ios::binary | ios::in);
	sp2 = SensorPair::loadSensorPair(input, sensors, nullptr);
	input.close();

	EXPECT_EQ(sp1->_uuid, sp2->_uuid);
	EXPECT_EQ(sp1->_vthreshold, sp2->_vthreshold);
	EXPECT_EQ(sp1->_sensor_i, sp2->_sensor_i);
	EXPECT_EQ(sp1->_sensor_j, sp2->_sensor_j);
}

AttrSensorPairSavingLoading::AttrSensorPairSavingLoading() {
	fileName = "attr_sensor_pair.uma";
	const string uuid = "attr_sensor_0";

	pair<string, string> p1 = { "s1", "cs1" };
	pair<string, string> p2 = { "s2", "cs2" };

	as1 = new AttrSensor("attr_sensor0", nullptr, 0, true, 0.12);
	as2 = new AttrSensor("attr_sensor1", nullptr, 2, false, 0.23);

	asp1 = new AttrSensorPair(nullptr, as1, as2, 0.98, true);
}

AttrSensorPairSavingLoading::~AttrSensorPairSavingLoading() {
	SysUtil::UMARemove(fileName);

	delete asp1, asp2;
	delete as1, as2;
}

void AttrSensorPairSavingLoading::savingAndLoading() {
	ofstream output;
	output.open(fileName, ios::binary | ios::out);
	asp1->saveAttrSensorPair(output);
	output.close();

	ifstream input;
	input.open(fileName, ios::binary | ios::in);
	asp2 = AttrSensorPair::loadAttrSensorPair(input, as1, as2, true, nullptr);
	input.close();

	EXPECT_EQ(asp1->_uuid, asp2->_uuid);
	EXPECT_EQ(asp1->_attrSensorI, asp2->_attrSensorI);
	EXPECT_EQ(asp1->_attrSensorJ, asp2->_attrSensorJ);
	EXPECT_EQ(asp1->_vw, asp2->_vw);
}

SnapshotSavingLoading::SnapshotSavingLoading() {
	s1FileName = "s1.uma";
	s2FileName = "s2.uma";
	s3FileName = "s3.uma";
	s4FileName = "s4.uma";

	agent1 = new Agent("agent1", nullptr);
	agent2 = new Agent("agent2", nullptr);
	s1 = new Snapshot("snapshot0", agent1);
	s2 = new SnapshotQualitative("snapshot1", agent1);
	s3 = new SnapshotEmpirical("snapshot2", agent1);
	s4 = new SnapshotDiscounted("snapshot3", agent1);
	s1->setInitialSize();
	s2->setInitialSize();
	s3->setInitialSize();
	s4->setInitialSize();
}

SnapshotSavingLoading::~SnapshotSavingLoading() {
	SysUtil::UMARemove(s1FileName);
	SysUtil::UMARemove(s2FileName);
	SysUtil::UMARemove(s3FileName);
	SysUtil::UMARemove(s4FileName);

	delete agent1, agent2;
	delete s1, s2, s3, s4, s5, s6, s7, s8;
}

void SnapshotSavingLoading::savingAndLoading() {
	ofstream output;
	output.open(s1FileName, ios::binary | ios::out);
	s1->saveSnapshot(output);
	output.close();

	output.open(s2FileName, ios::binary | ios::out);
	s2->saveSnapshot(output);
	output.close();

	output.open(s3FileName, ios::binary | ios::out);
	s3->saveSnapshot(output);
	output.close();

	output.open(s4FileName, ios::binary | ios::out);
	s4->saveSnapshot(output);
	output.close();

	ifstream input;
	input.open(s1FileName, ios::binary | ios::in);
	s5 = Snapshot::loadSnapshot(input, agent2);
	input.close();

	input.open(s2FileName, ios::binary | ios::in);
	s6 = Snapshot::loadSnapshot(input, agent2);
	input.close();

	input.open(s3FileName, ios::binary | ios::in);
	s7 = Snapshot::loadSnapshot(input, agent2);
	input.close();

	input.open(s4FileName, ios::binary | ios::in);
	s8 = Snapshot::loadSnapshot(input, agent2);
	input.close();

	EXPECT_EQ(s1->_type, s5->_type);
	EXPECT_EQ(s2->_type, s6->_type);
	EXPECT_EQ(s3->_type, s7->_type);
	EXPECT_EQ(s4->_type, s8->_type);

	EXPECT_EQ(s1->_uuid, s5->_uuid);
	EXPECT_EQ(s2->_uuid, s6->_uuid);
	EXPECT_EQ(s3->_uuid, s7->_uuid);
	EXPECT_EQ(s4->_uuid, s8->_uuid);

	EXPECT_EQ(s1->_initialSize, s5->_initialSize);
	EXPECT_EQ(s2->_initialSize, s6->_initialSize);
	EXPECT_EQ(s3->_initialSize, s7->_initialSize);
	EXPECT_EQ(s4->_initialSize, s8->_initialSize);
}

AgentSavingLoading::AgentSavingLoading() {
	a1FileName = "a1.uma";
	a2FileName = "a2.uma";
	a3FileName = "a3.uma";
	a4FileName = "a4.uma";

	a1 = new Agent("agent1", nullptr);
	a2 = new AgentQualitative("agent2", nullptr);
	a3 = new AgentDiscounted("agent3", nullptr);
	a4 = new AgentEmpirical("agent4", nullptr);
}

AgentSavingLoading::~AgentSavingLoading() {
	SysUtil::UMARemove(a1FileName);
	SysUtil::UMARemove(a2FileName);
	SysUtil::UMARemove(a3FileName);
	SysUtil::UMARemove(a4FileName);

	delete a1, a2, a3, a4, a5, a6, a7, a8;
}

void AgentSavingLoading::savingAndLoading() {
	ofstream output;
	output.open(a1FileName, ios::binary | ios::out);
	a1->saveAgent(output);
	output.close();

	output.open(a2FileName, ios::binary | ios::out);
	a2->saveAgent(output);
	output.close();

	output.open(a3FileName, ios::binary | ios::out);
	a3->saveAgent(output);
	output.close();

	output.open(a4FileName, ios::binary | ios::out);
	a4->saveAgent(output);
	output.close();

	ifstream input;
	input.open(a1FileName, ios::binary | ios::in);
	a5 = Agent::loadAgent(input, nullptr);
	input.close();

	input.open(a2FileName, ios::binary | ios::in);
	a6 = Agent::loadAgent(input, nullptr);
	input.close();

	input.open(a3FileName, ios::binary | ios::in);
	a7 = Agent::loadAgent(input, nullptr);
	input.close();

	input.open(a4FileName, ios::binary | ios::in);
	a8 = Agent::loadAgent(input, nullptr);
	input.close();

	EXPECT_EQ(a1->_type, a5->_type);
	EXPECT_EQ(a2->_type, a6->_type);
	EXPECT_EQ(a3->_type, a7->_type);
	EXPECT_EQ(a4->_type, a8->_type);

	EXPECT_EQ(a1->_uuid, a5->_uuid);
	EXPECT_EQ(a2->_uuid, a6->_uuid);
	EXPECT_EQ(a3->_uuid, a7->_uuid);
	EXPECT_EQ(a4->_uuid, a8->_uuid);
}

ExperimentSavingLoading::ExperimentSavingLoading() {
	exp1 = new Experiment("experiment0");
}

ExperimentSavingLoading::~ExperimentSavingLoading() {
	SysUtil::UMARemove(exp1->getUUID() + ".uma");

	delete exp1, exp2;
}

void ExperimentSavingLoading::savingAndLoading() {
	exp1->saveExperiment();

	exp2 = Experiment::loadExperiment(exp1->getUUID());

	EXPECT_EQ(exp1->_uuid, exp2->_uuid);
	EXPECT_EQ(exp1->_agents.size(), exp2->_agents.size());
}

UMASavingLoading::UMASavingLoading() {
	exp1 = new Experiment("uma_test");
	Agent *a1 = exp1->createAgent("agent1", UMA_AGENT::AGENT_QUALITATIVE);
	Agent *a2 = exp1->createAgent("agent2", UMA_AGENT::AGENT_DISCOUNTED);

	Snapshot *s1 = a1->createSnapshot("snapshot1");
	Snapshot *s2 = a1->createSnapshot("snapshot2");
	Snapshot *s3 = a2->createSnapshot("snapshot3");
	Snapshot *s4 = a2->createSnapshot("snapshot4");

	std::pair<string, string> sList1 = { "s1", "cs1" };
	std::pair<string, string> sList2 = { "s2", "cs2" };
	std::pair<string, string> sList3 = { "s3", "cs3" };

	vector<double> diag1 = { 0.4, 0.6 };
	vector<double> diag2 = { 0.3, 0.7 };
	vector<vector<double>> w1 = { {0.2, 0.2, 0,0, 0.6} }, w2 = { {0.2, 0.1, 0.3, 0.4}, {0.0, 0.5, 0.2, 0.3} };
	vector<vector<double>> w3 = { {0,9, 0.1, 0, 0}, {0.2, 0.3, 0.1, 0.4}, {0.7, 0.2, 0.05, 0.05} };
	vector<vector<bool>> b1 = { {true, false, false, true} }, b2 = { {false, false, true, true}, {true, true, false, false} };
	vector<vector<bool>> b3 = { {false, false, false, false}, {true, false, true, false}, {false, true, false, false} };
	s1->createSensor(sList1, diag1, w1, b1);
	s1->createSensor(sList3, diag2, w2, b2);
	s2->createSensor(sList2, diag2, w1, b1);
	s3->createSensor(sList1, diag1, w1, b1);
	s3->createSensor(sList2, diag2, w2, b2);
	s3->createSensor(sList3, diag1, w3, b3);

	s1->setInitialSize();
	s2->setInitialSize();
	s3->setInitialSize();
	s4->setInitialSize();
}

UMASavingLoading::~UMASavingLoading() {
	SysUtil::UMARemove(exp1->getUUID() + ".uma");

	delete exp1, exp2;
}

void UMASavingLoading::savingAndLoading() {
	exp1->saveExperiment();

	exp2 = Experiment::loadExperiment(exp1->getUUID());

	EXPECT_EQ(exp1->_uuid, exp2->_uuid);
	for (auto agentIt = exp1->_agents.begin(); agentIt != exp1->_agents.end(); ++agentIt) {
		string agentName = agentIt->first;
		Agent *agent1 = exp1->_agents[agentName], *agent2 = exp2->_agents[agentName];
		EXPECT_EQ(agent1->_type, agent2->_type);
		EXPECT_EQ(agent1->_uuid, agent2->_uuid);

		for (auto snapshotIt = agent1->_snapshots.begin(); snapshotIt != agent1->_snapshots.end(); ++snapshotIt) {
			string snapshotName = snapshotIt->first;
			Snapshot *snapshot1 = agent1->_snapshots[snapshotName], *snapshot2 = agent2->_snapshots[snapshotName];
			EXPECT_EQ(snapshot1->_type, snapshot2->_type);
			EXPECT_EQ(snapshot1->_uuid, snapshot2->_uuid);
			EXPECT_EQ(snapshot1->_initialSize, snapshot2->_initialSize);

			for (int i = 0; i < snapshot1->_sensors.size(); ++i) {
				Sensor *sensor1 = snapshot1->_sensors[i], *sensor2 = snapshot2->_sensors[i];
				EXPECT_EQ(sensor1->_uuid, sensor2->_uuid);
				EXPECT_EQ(sensor1->_idx, sensor2->_idx);
				EXPECT_EQ(sensor1->_amper, sensor2->_amper);

				EXPECT_EQ(sensor1->_m->_uuid, sensor2->_m->_uuid);
				EXPECT_EQ(sensor1->_m->_idx, sensor2->_m->_idx);
				EXPECT_EQ(sensor1->_m->_isOriginPure, sensor2->_m->_isOriginPure);
				EXPECT_EQ(sensor1->_m->_vdiag, sensor2->_m->_vdiag);
				EXPECT_EQ(sensor1->_m->_vdiag_, sensor2->_m->_vdiag_);

				EXPECT_EQ(sensor1->_cm->_uuid, sensor2->_cm->_uuid);
				EXPECT_EQ(sensor1->_cm->_idx, sensor2->_cm->_idx);
				EXPECT_EQ(sensor1->_cm->_isOriginPure, sensor2->_cm->_isOriginPure);
				EXPECT_EQ(sensor1->_cm->_vdiag, sensor2->_cm->_vdiag);
				EXPECT_EQ(sensor1->_cm->_vdiag_, sensor2->_cm->_vdiag_);
			}

			for (int i = 0; i < snapshot1->_sensorPairs.size(); ++i) {
				SensorPair *sensorPair1 = snapshot1->_sensorPairs[i], *sensorPair2 = snapshot2->_sensorPairs[i];
				EXPECT_EQ(sensorPair1->_uuid, sensorPair2->_uuid);
				EXPECT_EQ(sensorPair1->_vthreshold, sensorPair2->_vthreshold);
				
				EXPECT_EQ(sensorPair1->mij->_uuid, sensorPair2->mij->_uuid);
				EXPECT_EQ(sensorPair1->mij->_vw, sensorPair2->mij->_vw);

				EXPECT_EQ(sensorPair1->mi_j->_uuid, sensorPair2->mi_j->_uuid);
				EXPECT_EQ(sensorPair1->mi_j->_vw, sensorPair2->mi_j->_vw);

				EXPECT_EQ(sensorPair1->m_ij->_uuid, sensorPair2->m_ij->_uuid);
				EXPECT_EQ(sensorPair1->m_ij->_vw, sensorPair2->m_ij->_vw);

				EXPECT_EQ(sensorPair1->m_i_j->_uuid, sensorPair2->m_i_j->_uuid);
				EXPECT_EQ(sensorPair1->m_i_j->_vw, sensorPair2->m_i_j->_vw);
			}
		}
	}
}