from multiprocessing import Pool
#from multiprocessing import Process
import UMA_sniffy
import sys
import os
import time
import cPickle

def experiment(args):
    return UMA_sniffy.start_experiment(*args)

if __name__ == "__main__":
    PROCESSES=3

    clk=time.clock()

    preamble={
        'env_length':20,
        'total_cycles':300,
        'burn_in_cycles':200,
        'name':sys.argv[1],
        'ex_dataQ':False,
        'agent_dataQ':False,
        'mids_to_record':['count','dist','sig'],
        'Nruns':int(sys.argv[2])
        }

    #Make a preamble file
    os.mkdir(preamble['name'])
    preamblef=open(".\\"+preamble['name']+"\\"+preamble['name']+".pre","wb")
    cPickle.dump(preamble,preamblef,protocol=cPickle.HIGHEST_PROTOCOL)
    preamblef.close()
    
    #construct a generator for first input of UMA_sniffy.start_experiment (preamble)
    def run_params():
        for ind in xrange(preamble['Nruns']):
            yield preamble,preamble['name']+"_"+str(ind)

    #run the experiment
    p=Pool(processes=PROCESSES)
    p.map(experiment,run_params(),chunksize=PROCESSES)
    p.close()
    p.join()

    print "All runs are done!\n"
    print "Elapsed time: "+str(time.clock()-clk)+"\n"

