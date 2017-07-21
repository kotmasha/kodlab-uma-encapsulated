#include "logging.h"

logging::logging(){}

logging::logging(const logging &l){}

logging::logging(string filename, string classname){
	_filename = filename;
	_classname = parse_class_name(classname);
	_output = new ofstream(filename);
}

string logging::parse_class_name(string classname){
	classname = classname.substr(classname.find("class") + 6, classname.size() - 6);
	return classname;
}

void logging::operator<<(string info){
	if(!_active) return;
	string output_info = "[" + _classname + "]:" + _level + " " + info + "\n";
	_output->write(output_info.c_str(), output_info.size() * sizeof(char));
	_output->flush();
}

void logging::operator<<(wstring info) {
	if (!_active) return;
	std::string s_info(info.begin(), info.end());
	string output_info = "[" + _classname + "]:" + _level + " " + s_info + "\n";
	_output->write(output_info.c_str(), output_info.size() * sizeof(char));
	_output->flush();
}
