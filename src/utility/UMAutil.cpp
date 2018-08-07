#include "UMAutil.h"
#include "UMAException.h"

/*
##################SysUtil####################
*/
/*
input: string path
*/
string SysUtil::UMAMkdir(string path) {
#if defined(_WIN64)
	_mkdir(path.c_str());
#else 
	mkdir(path.c_str(), 0777);
#endif
	return path;
}
/*
##################SysUtil####################
*/


/*
##################StrUtil####################
*/
/*
input: string level
output: int level
*/
int StrUtil::stringToLogLevel(const string &s) {
	if (s == "ERROR") return 0;
	if (s == "WARN") return 1;
	if (s == "INFO") return 2;
	if (s == "DEBUG") return 3;
	if (s == "VERBOSE") return 4;
}

vector<std::pair<string, string>> StrUtil::string2dToString1dPair(const vector<vector<string>> &pairs) {
	vector<std::pair<string, string> > results;
	for (int i = 0; i < pairs.size(); ++i) {
		const vector<string> p = pairs[i];
		if (p.size() != 2) {
			throw UMAException("The " + to_string(i) + "th vector is not a size 2 vector, cannot convert to pairs!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
		}
		results.push_back(pair<string, string>(p[0], p[1]));
	}
	return results;
}

/*
##################StrUtil####################
*/

/*
##################SignalUtil####################
*/

/*
This function is converting the list, from bool to int
Input: a bool signal
Output: a int vector
*/
const vector<int> SignalUtil::boolSignalToIntIdx(const vector<bool> &list){
	vector<int> convertedList;
	for (int i = 0; i < list.size(); ++i) {
		if (list[i]) convertedList.push_back(i);
	}
	return convertedList;
}

/*
This function is converting the idx, from int to bool list
Input: a int vector, size of output bool signal
Output: a bool signal
*/
const vector<bool> SignalUtil::intIdxToBoolSignal(const vector<int> &idx, int size) {
	vector<bool> convertedList(size, false);
	for (int i = 0; i < idx.size(); ++i) {
		convertedList[idx[i]] = true;
	}
	return convertedList;
}

/*
This function is converting the attr_sensor signal to sensor signal, any of the 2 attr_sensor signal is true will lead to true in the sensor signal
Input: attr_sensor signal
Output: sensor signal
*/
const vector<bool> SignalUtil::attrSensorToSensorSignal(const vector<bool> &signal) {
	vector<bool> result;
	for (int i = 0; i < signal.size() / 2; ++i) {
		if (signal[2 * i] || signal[2 * i + 1]) result.push_back(true);
		else result.push_back(false);
	}
	return result;
}

/*
This function is trimming the signal value, removing all false value in the BACK of the signal only
Input: signal to trim
*/
const vector<bool> SignalUtil::trimSignal(const vector<bool> &signal) {
	vector<bool> result = signal;
	while (!result.empty()) {
		if (false == result.back()) result.pop_back();
		else break;
	}
	return result;
}
/*
##################SignalUtil####################
*/

/*
This function is finding the target position in a given input, based on integer
It will find the position of the index if exist, or the position that is in the gap of two value
ex: [1,2,3],1 = 0
ex: [1,2,4],3 = 1
Input: input int signal, target int to find
Output: the index of target
*/
int ArrayUtil::findIdxInSortedArray(const vector<int> &input, int target) {
	if (input.empty()) return -1;
	if (target < input[0]) return -1;
	if (target == input[0]) return 0;
	if (target >= input.back()) return input.size() - 1;
	int startIdx = 0;
	int endIdx = input.size();
	int midIdx = (startIdx + endIdx) / 2;
	while (true) {
		if (input[midIdx] == target) return midIdx;
		if (input[midIdx] > target) {
			if (input[midIdx - 1] < target) return midIdx - 1;
			endIdx = midIdx;
		}
		else {
			if (input[midIdx + 1] > target) return midIdx;
			startIdx = midIdx;
		}
		midIdx = (startIdx + endIdx) / 2;
	}

}