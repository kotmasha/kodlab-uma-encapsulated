g++ -fPIC -std=c++11 -c Agent.cpp -o Agent.o;
g++ -fPIC -std=c++11 -c worker.cpp -o worker.o;
g++ -fPIC -std=c++11 -c logging.cpp -o logging.o;
swig -c++ -python -o UMA_NEW_wrap.cpp UMA_NEW.i;
gcc -fPIC -c UMA_NEW_wrap.cpp -o UMA_NEW_wrap.o -I/usr/include/python2.7;
nvcc -shared -Xcompiler -fPIC kernel.cu Agent.o worker.o logging.o UMA_NEW_wrap.o -o _UMA_NEW.so;