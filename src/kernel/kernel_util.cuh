#ifndef _KERNEL_UTIL
#define _KERNEL_UTIL

namespace kernel_util {
	void alltrue(bool *b, int size);
	void allfalse(bool *b, int size);
	void bool2int(bool *b, int *i, int size);
	void bool2double(bool *b, double *d, int size);
	void conjunction(bool *b1, bool *b2, int size);
	void disjunction(bool *b1, bool *b2, int size);
	void subtraction(bool *b1, bool *b2, int size);
	void negateConjunctionStar(bool *b1, bool *b2, int size);
	void ConjunctionStar(bool *b1, bool *b2, int size);
	void up2down(bool *b1, bool *b2, int size);
	double sum(double *d, int size);
	void initMaskSignal(bool *b, int initSize, int size);
}

#endif