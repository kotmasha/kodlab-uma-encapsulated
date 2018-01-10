#ifndef _KERNEL_UTIL
#define _KERNEL_UTIL

namespace kernel_util {
	void alltrue(bool *b, int size);
	void allfalse(bool *b, int size);
	void bool2int(bool *b, int *i, int size);
	void conjunction(bool *b1, bool *b2, int size);
	void disjunction(bool *b1, bool *b2, int size);
	void subtraction(bool *b1, bool *b2, int size);
	void negate_conjunction_star(bool *b1, bool *b2, int size);
	void conjunction_star(bool *b1, bool *b2, int size);
	void up2down(bool *b1, bool *b2, int size);
}

#endif