%module UMA

%include "std_vector.i"
%include "std_string.i"
%include "std_map.i"

%{
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "Measurable.h"
#include "UMATest.h"
%}

namespace std {
   %template(IntVector) std::vector<int>;
   %template(DoubleVector) std::vector<double>;
   %template(StringVector) std::vector<string>;
   %template(ConstCharVector) std::vector<const char*>;
   %template(BoolVector) std::vector<bool>;
   %template(IntVector2D) std::vector<std::vector<int>>;
   %template(DoubleVector2D) std::vector<std::vector<double>>;
   %template(StringVector2D) std::vector<std::vector<string>>;
   %template(ConstCharVector2D) std::vector<std::vector<const char*>>;
   %template(BoolVector2D) std::vector<std::vector<bool>>;
}

%include "Agent.h"
%include "Snapshot.h"
%include "Sensor.h"
%include "SensorPair.h"
%include "Measurable.h"
%include "UMATest.h"