#ifndef _DATA_UTIL_
#define _DATA_UTIL_

namespace data_util{
	void boolD2D(bool *from, bool *to, int size, int from_offset=0, int to_offset=0);
	void boolD2H(bool *from, bool *to, int size, int from_offset = 0, int to_offset = 0);
	void boolH2D(bool *from, bool *to, int size, int from_offset = 0, int to_offset = 0);
	void boolH2H(bool *from, bool *to, int size, int from_offset = 0, int to_offset = 0);

	void doubleD2D(double *from, double *to, int size, int from_offset = 0, int to_offset = 0);
	void doubleD2H(double *from, double *to, int size, int from_offset = 0, int to_offset = 0);
	void doubleH2D(double *from, double *to, int size, int from_offset = 0, int to_offset = 0);
	void doubleH2H(double *from, double *to, int size, int from_offset = 0, int to_offset = 0);

	void intD2D(int *from, int *to, int size, int from_offset = 0, int to_offset = 0);
	void intD2H(int *from, int *to, int size, int from_offset = 0, int to_offset = 0);
	void intH2D(int *from, int *to, int size, int from_offset = 0, int to_offset = 0);
	void intH2H(int *from, int *to, int size, int from_offset = 0, int to_offset = 0);

	void floatD2D(float *from, float *to, int size, int from_offset = 0, int to_offset = 0);
	void floatD2H(float *from, float *to, int size, int from_offset = 0, int to_offset = 0);
	void floatH2D(float *from, float *to, int size, int from_offset = 0, int to_offset = 0);
	void floatH2H(float *from, float *to, int size, int from_offset = 0, int to_offset = 0);

	void dev_bool(bool *&dst, int size);
	void dev_int(int *&dst, int size);
	void dev_double(double *&dst, int size);
	void dev_float(float *&dst, int size);

	void dev_init(bool *ptr, int size);
	void dev_init(double *ptr, int size);
	void dev_init(int *ptr, int size);
	void dev_init(float *ptr, int size);

	void dev_free(bool *ptr);
	void dev_free(double *ptr);
	void dev_free(int *ptr);
	void dev_free(float *ptr);
}

#endif