#include "worker.h"

worker::worker(){
}

worker::worker(string sensor_name1,string sensor_name2,int sensor_id1,int sensor_id2){
	this->sensor_name1=sensor_name1;
	this->sensor_name2=sensor_name2;
	this->sensor_id1=sensor_id1;
	this->sensor_id2=sensor_id2;
}

void worker::add_time(){
	t++;
}

void worker::reset_time(){
	t=0;
}