#include "UMAutil.h"
#include "UMAException.h"

/*
##################SysUtil####################
*/
/*
input: string path
*/
string SysUtil::UMA_mkdir(string path) {
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
int StrUtil::string_to_log_level(const string &s) {
	if (s == "ERROR") return 0;
	if (s == "WARN") return 1;
	if (s == "INFO") return 2;
	if (s == "DEBUG") return 3;
	if (s == "VERBOSE") return 4;
}

vector<std::pair<string, string>> StrUtil::string2d_to_string1d_pair(const vector<vector<string>> &pairs) {
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
const vector<int> SignalUtil::bool_signal_to_int_idx(const vector<bool> &list){
	vector<int> converted_list;
	for (int i = 0; i < list.size(); ++i) {
		if (list[i]) converted_list.push_back(i);
	}
	return converted_list;
}

/*
This function is converting the measurable signal to sensor signal, any of the 2 measurable signal is true will lead to true in the sensor signal
Input: measurable signal
Output: sensor signal
*/
const vector<bool> SignalUtil::measurable_signal_to_sensor_signal(const vector<bool> &signal) {
	vector<bool> result;
	for (int i = 0; i < signal.size() / 2; ++i) {
		if (signal[2 * i] || signal[2 * i + 1]) result.push_back(true);
		else result.push_back(false);
	}
	return result;
}
/*
##################SignalUtil####################
*/