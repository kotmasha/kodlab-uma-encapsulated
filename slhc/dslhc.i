%module dslhc

%include "std_vector.i"
%include "std_string.i"
%{
#include "../../Blocks/World.h"
#include "../../Blocks/Pair.h"
#include "../../Blocks/DataManager.h"
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

%include "../../Blocks/World.h"
%include "../../Blocks/Pair.h"
%include "../../Blocks/DataManager.h"