%module UMA_NEW

%include "std_vector.i"
%include "std_string.i"
%include "std_map.i"
%{
#include "../../UMA_NEW/Agent.h"
#include "../../UMA_NEW/logging.h"
#include "../../UMA_NEW/AgentDM.h"
#include "../../UMA_NEW/UMATest.h"
%}

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(ConstCharVector) vector<const char*>;
   %template(BoolVector) vector<bool>;
   %template(IntVector2D) vector<vector<int>>;
   %template(DoubleVector2D) vector<vector<double>>;
   %template(StringVector2D) vector<vector<string>>;
   %template(ConstCharVector2D) vector<vector<const char*>>;
   %template(BoolVector2D) vector<vector<bool>>;
}

%include "../../UMA_NEW/Agent.h"
%include "../../UMA_NEW/logging.h"
%include "../../UMA_NEW/AgentDM.h"
%include "../../UMA_NEW/UMATest.h"