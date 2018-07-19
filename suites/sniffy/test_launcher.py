from multiprocessing import Pool
#from multiprocessing import Process
import UMA_sniffy
import sys
import os
import yaml
import time
import cPickle

DEPLOY_YML = 'deploy.yml'
UMA_HOME = os.path.dirname(os.path.dirname(os.getcwd()))

def experiment(args):
    return UMA_sniffy.start_experiment(*args)

def mapping_port(idx, port, instance):
    print str(int(port) + idx % instance)
    return str(int(port) + idx % instance)

if __name__ == "__main__":
    PROCESSES=3

    with open(os.path.join(UMA_HOME, 'deployment', DEPLOY_YML), 'r') as f:
        info = yaml.load(f)
        instance = info['Cluster']['instance']
        base_url = info['Cluster']['base_url']
        port = int(info['Cluster']['port'])

    clk=time.clock()

    def get_preamble(ind):
        return {
            'env_length':20,
            'total_cycles':300,
            'burn_in_cycles':200,
            'name':sys.argv[1],
            'ex_dataQ':False,
            'agent_dataQ':False,
            'mids_to_record':['count','dist','sig'],
            'host': base_url,
            'port': mapping_port(ind, port, instance),
            'Nruns':int(sys.argv[2])
        }

    #Make a preamble file
    os.mkdir(sys.argv[1])
    preamblef=open(".\\"+sys.argv[1]+"\\"+sys.argv[1]+".pre","wb")
    cPickle.dump(get_preamble(0),preamblef,protocol=cPickle.HIGHEST_PROTOCOL)
    preamblef.close()
    
    #construct a generator for first input of UMA_sniffy.start_experiment (preamble)
    def run_params():
        for ind in xrange(get_preamble(0)['Nruns']):
            yield get_preamble(ind), sys.argv[1]+"_"+str(ind)

    #run the experiment
    p=Pool(processes=PROCESSES)
    p.map(experiment,run_params(),chunksize=PROCESSES)
    p.close()
    p.join()

    print "All runs are done!\n"
    print "Elapsed time: "+str(time.clock()-clk)+"\n"

