#include "UMACoreTestFixture.h"

extern int ind(int row, int col);
extern int compi(int x);

WorldTestFixture::WorldTestFixture() {}

WorldTestFixture::~WorldTestFixture() {}

AmperAndTestFixture::AmperAndTestFixture(){
	snapshot = new Snapshot("snapshot", "");
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
	snapshot->add_sensor(p0, diag, w0, b0);
	snapshot->add_sensor(p1, diag, w1, b1);
	snapshot->add_sensor(p2, diag, w2, b2);
	snapshot->add_sensor(p3, diag, w3, b3);
}

AmperAndTestFixture::~AmperAndTestFixture() {
	delete snapshot;
}

vector<vector<double>> AmperAndTestFixture::test_amper_and(int mid1, int mid2, bool merge, std::pair<string, string> &p) {
	snapshot->amperand(mid1, mid2, merge, p);

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
	snapshot = new Snapshot("snapshot", "");
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
	Sensor *s0 = snapshot->add_sensor(p0, diag, w0, b0);
	Sensor *s1 = snapshot->add_sensor(p1, diag, w1, b1);
	Sensor *s2 = snapshot->add_sensor(p2, diag, w2, b2);
	Sensor *s3 = snapshot->add_sensor(p3, diag, w3, b3);
	
	snapshot->getAttrSensor(0)->_vdiag = 0.2;
	snapshot->getAttrSensor(1)->_vdiag = 0.8;
	snapshot->getAttrSensor(2)->_vdiag = 0.4;
	snapshot->getAttrSensor(3)->_vdiag = 0.6;
	snapshot->getAttrSensor(4)->_vdiag = 0.6;
	snapshot->getAttrSensor(5)->_vdiag = 0.4;
	snapshot->getAttrSensor(6)->_vdiag = 0.8;
	snapshot->getAttrSensor(7)->_vdiag = 0.2;
}

GenerateDelayedWeightsTestFixture::~GenerateDelayedWeightsTestFixture() {
	delete snapshot;
}

vector<vector<double>> GenerateDelayedWeightsTestFixture::test_generate_delayed_weights(int mid, bool merge, const std::pair<string, string> &id_pair, vector<bool> &observe) {
	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		*(snapshot->getAttrSensor(i)->_observe) = observe[i];
		*(snapshot->getAttrSensor(i)->_observe_) = observe[i];
	}
	snapshot->generate_delayed_weights(mid, merge, id_pair);

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

AmperTestFixture::AmperTestFixture() {
	snapshot = new Snapshot("snapshot", "");
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
	snapshot->add_sensor(p0, diag, w0, b0);
	snapshot->add_sensor(p1, diag, w1, b1);
	snapshot->add_sensor(p2, diag, w2, b2);
	snapshot->add_sensor(p3, diag, w3, b3);
}

AmperTestFixture::~AmperTestFixture() {
	delete snapshot;
}

vector<vector<double>> AmperTestFixture::test_amper(const vector<int> &list, const std::pair<string, string> &uuid) {
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
	dm = new DataManager("");

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

	sensors.push_back(new Sensor(sensor1, 1.0, 0));
	sensors.push_back(new Sensor(sensor2, d1, 1));
	sensors.push_back(new Sensor(sensor3, 1.0, 2));
	sensors.push_back(new Sensor(sensor4, 1.0, 3));
	sensorPairs.push_back(new SensorPair(sensors[0], sensors[0], 0.125, w0[0], b0[0]));
	sensorPairs.push_back(new SensorPair(sensors[1], sensors[0], 0.25, 1.0));
	sensorPairs.push_back(new SensorPair(sensors[1], sensors[1], 0.25, 1.0));
	sensorPairs.push_back(new SensorPair(sensors[2], sensors[0], 0.125, w2[0], b2[0]));
	sensorPairs.push_back(new SensorPair(sensors[2], sensors[1], 0.125, w2[1], b2[1]));
	sensorPairs.push_back(new SensorPair(sensors[2], sensors[2], 0.125, w2[2], b2[2]));
	sensorPairs.push_back(new SensorPair(sensors[3], sensors[0], 0.5, 1.0));
	sensorPairs.push_back(new SensorPair(sensors[3], sensors[1], 0.5, 1.0));
	sensorPairs.push_back(new SensorPair(sensors[3], sensors[2], 0.5, 1.0));
	sensorPairs.push_back(new SensorPair(sensors[3], sensors[3], 0.5, 1.0));

	sensors[0]->_m->_vobserve = true; sensors[0]->_m->_vobserve_ = true;
	sensors[0]->_cm->_vobserve = false; sensors[0]->_cm->_vobserve_ = false;
	sensors[1]->_m->_vobserve = false; sensors[1]->_m->_vobserve_ = false;
	sensors[1]->_cm->_vobserve = true; sensors[1]->_cm->_vobserve_ = true;
	sensors[2]->_m->_vobserve = false; sensors[2]->_m->_vobserve_ = false;
	sensors[2]->_cm->_vobserve = true; sensors[2]->_cm->_vobserve_ = true;
	sensors[3]->_m->_vobserve = true; sensors[3]->_m->_vobserve_ = true;
	sensors[3]->_cm->_vobserve = false; sensors[3]->_cm->_vobserve_ = false;

	double total = 1.0;
	dm->reallocate_memory(total, 4);
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

void UMACoreDataFlowTestFixture::test_uma_core_dataflow(int start_idx, int end_idx) {
	dm->create_sensors_to_arrays_index(start_idx, end_idx, sensors);
	dm->create_sensor_pairs_to_arrays_index(start_idx, end_idx, sensorPairs);

	for (int i = start_idx; i < end_idx; ++i) {
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
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		int idx_i = 2 * sensorPairs[i]->_sensor_i->_idx;
		int idx_j = 2 * sensorPairs[i]->_sensor_j->_idx;
		EXPECT_EQ(sensorPairs[i]->mij->_w, dm->h_weights + ind(idx_i, idx_j));
		EXPECT_EQ(sensorPairs[i]->mi_j->_w, dm->h_weights + ind(idx_i, idx_j + 1));
		EXPECT_EQ(sensorPairs[i]->m_ij->_w, dm->h_weights + ind(idx_i + 1, idx_j));
		EXPECT_EQ(sensorPairs[i]->m_i_j->_w, dm->h_weights + ind(idx_i + 1, idx_j + 1));

		EXPECT_EQ(sensorPairs[i]->mij->_d, dm->h_dirs + ind(idx_i, idx_j));
		EXPECT_EQ(sensorPairs[i]->mi_j->_d, dm->h_dirs + ind(idx_i, idx_j + 1));
		EXPECT_EQ(sensorPairs[i]->m_ij->_d, dm->h_dirs + ind(idx_i + 1, idx_j));
		EXPECT_EQ(sensorPairs[i]->m_i_j->_d, dm->h_dirs + ind(idx_i + 1, idx_j + 1));

		EXPECT_EQ(sensorPairs[i]->_threshold, dm->h_thresholds + ind(idx_i / 2, idx_j / 2));
	}

	dm->copy_sensors_to_arrays(start_idx, end_idx, sensors);
	dm->copy_sensor_pairs_to_arrays(start_idx, end_idx, sensorPairs);
	dm->copy_arrays_to_sensors(start_idx, end_idx, sensors);
	dm->copy_arrays_to_sensor_pairs(start_idx, end_idx, sensorPairs);

	for (int i = start_idx; i < end_idx; ++i) {
		EXPECT_EQ(diag[2 * i], sensors[i]->_m->_vdiag);
		EXPECT_EQ(diag[2 * i + 1], sensors[i]->_cm->_vdiag);
		EXPECT_EQ(diag[2 * i], sensors[i]->_m->_vdiag_);
		EXPECT_EQ(diag[2 * i + 1], sensors[i]->_cm->_vdiag_);
		EXPECT_EQ(observe[2 * i], sensors[i]->_m->_vobserve);
		EXPECT_EQ(observe[2 * i], sensors[i]->_m->_vobserve_);
		EXPECT_EQ(observe[2 * i + 1], sensors[i]->_cm->_vobserve);
		EXPECT_EQ(observe[2 * i + 1], sensors[i]->_cm->_vobserve_);
	}

	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		int idx_i = 2 * sensorPairs[i]->_sensor_i->_idx;
		int idx_j = 2 * sensorPairs[i]->_sensor_j->_idx;
		EXPECT_DOUBLE_EQ(weights[ind(idx_i, idx_j)], sensorPairs[i]->mij->_vw);
		EXPECT_DOUBLE_EQ(weights[ind(idx_i, idx_j + 1)], sensorPairs[i]->mi_j->_vw);
		EXPECT_DOUBLE_EQ(weights[ind(idx_i + 1, idx_j)], sensorPairs[i]->m_ij->_vw);
		EXPECT_DOUBLE_EQ(weights[ind(idx_i + 1, idx_j + 1)], sensorPairs[i]->m_i_j->_vw);

		EXPECT_EQ(dirs[ind(idx_i, idx_j)], sensorPairs[i]->mij->_vd);
		EXPECT_EQ(dirs[ind(idx_i, idx_j + 1)], sensorPairs[i]->mi_j->_vd);
		EXPECT_EQ(dirs[ind(idx_i + 1, idx_j)], sensorPairs[i]->m_ij->_vd);
		EXPECT_EQ(dirs[ind(idx_i + 1, idx_j + 1)], sensorPairs[i]->m_i_j->_vd);

		EXPECT_DOUBLE_EQ(thresholds[ind(idx_i / 2, idx_j / 2)], sensorPairs[i]->_vthreshold);
	}
}