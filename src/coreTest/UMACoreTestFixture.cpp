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

	EXPECT_EQ(as1->_uuid, as2->_uuid);
	EXPECT_EQ(as1->_idx, as2->_idx);
	EXPECT_EQ(as1->_isOriginPure, as2->_isOriginPure);
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

	EXPECT_EQ(asp1->_uuid, asp2->_uuid);
	EXPECT_EQ(asp1->_attrSensorI, asp2->_attrSensorI);
	EXPECT_EQ(asp1->_attrSensorJ, asp2->_attrSensorJ);
	EXPECT_EQ(asp1->_vw, asp2->_vw);
}
//TODO add test for other type of snapshot
SnapshotSavingLoading::SnapshotSavingLoading() {
	fileName = "snapshot.uma";

	agent1 = new Agent("agent1", nullptr);
	agent2 = new Agent("agent2", nullptr);
	snapshot1 = new Snapshot("snapshot0", agent1, UMA_SNAPSHOT::SNAPSHOT_STATIONARY);
}

SnapshotSavingLoading::~SnapshotSavingLoading() {
	SysUtil::UMARemove(fileName);

	delete agent1, agent2;
}
//TODO test duplicate
void SnapshotSavingLoading::savingAndLoading() {
	ofstream output;
	output.open(fileName, ios::binary | ios::out);
	snapshot1->saveSnapshot(output);
	output.close();

	ifstream input;
	input.open(fileName, ios::binary | ios::in);
	snapshot2 = Snapshot::loadSnapshot(input, agent2, UMA_SNAPSHOT::SNAPSHOT_STATIONARY);

	EXPECT_EQ(snapshot1->_uuid, snapshot2->_uuid);
	EXPECT_EQ(snapshot1->_type, snapshot2->_type);
	EXPECT_EQ(snapshot1->_sensors.size(), snapshot2->_sensors.size());
	EXPECT_EQ(snapshot1->_sensorPairs.size(), snapshot2->_sensorPairs.size());
}

AgentSavingLoading::AgentSavingLoading() {
	fileName = "agent.uma";

	agent1 = new Agent("agent1", nullptr);
}

AgentSavingLoading::~AgentSavingLoading() {
	SysUtil::UMARemove(fileName);

	delete agent1, agent2;
}

void AgentSavingLoading::savingAndLoading() {
	ofstream output;
	output.open(fileName, ios::binary | ios::out);
	agent1->saveAgent(output);
	output.close();

	ifstream input;
	input.open(fileName, ios::binary | ios::in);
	agent2 = Agent::loadAgent(input, nullptr);

	EXPECT_EQ(agent1->_uuid, agent2->_uuid);
	EXPECT_EQ(agent1->_snapshots.size(), agent2->_snapshots.size());
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