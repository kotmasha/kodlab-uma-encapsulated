from multiprocessing import Pool
from multiprocessing import Process
from UMA_sniffy import *
import sys

if __name__ == "__main__":
    NUM = 40
    test_name = ['test%s' % i for i in xrange(NUM)]
    test_x = [20 for i in xrange(NUM)]
    test_burn_in = [150 for i in xrange(NUM)]
    test_total = [300 for i in xrange(NUM)]

    p_list = []
    for i in xrange(len(test_name)):
        p = Process(target=main, args=(test_x[i], test_burn_in[i], test_total[i], test_name[i]))
        p_list.append(p)
        p.start()

    for p in p_list:
        p.join()

    print "all tests done!"
