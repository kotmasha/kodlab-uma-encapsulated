############ HE-based Control (HEC)
############ Main simulation module
from multiprocessing import Pool

#from functools import partial
import sys
from os import mkdir
import numpy as np
import cPickle

from module_scenario import *
import poison


N_CORES=2

def pickle_loader(pickled_file):
    try:
        while True:
            yield cPickle.load(pickled_file)
    except EOFError:
        pass

### INPUT: $dir_name$

### ASSUMPTIONS: The directory ./$dir_name$/ contains:
###     * A Python-pickled data file named "$dir_name$.dat":
###
###         ** each pickle in this file is a list of real numbers
###
###         ** the lengths of the pickled lists are constant
###
###     * A Python-pickled data file named "$dir_name$.pre", containing a Python dictionary:
###
###         ** 'RUNS'       -- the required number of runs of HEC on the data set;
###
###         ** 'SAMPLE_INITIAL' -- fraction of data set to be used for initial DB;
###
###         ** 'SAMPLE_FINAL' -- fraction of data set size to be reached for run to be stopped;
###                              ***HENCE ALL THE RUNS HAVE THE SAME LENGTH***
###
###         ** 'METRIC'     -- the metric/midpoint function to be used:
###             - 'ellone': self-explanatory
###             - 'ellone_Boolean': metric induced from ellone, but midpoint selected to be a
###                     a Boolean vector
###             - 'elltwo': self-explanatory.
###             - 'elltwo_normalized': assuming all data points are on the unit sphere, 
###                     bridging will also be confined to the unit sphere.
###
###         ** 'POISONm' -- poisoning method to be used:
###             - 'BridgeBest': optimal bridging of the MST with respect to selected 
###                     quality measure.
###             - 'BridgeRandom': randomized bridging of the MST.
###
###         ** 'POISONd' -- hierarchical dissimilarity to govern poisoning: 
###             - 'cluster_adjacency': ellone norm of the difference between cluster-adjacency
###                     matrices (if C=incidence matrix, then C*C^T is the adjacency matrix)
###             - 'worst_cut': Biggio's dissimilarity
###             - 'most_fusion': total difference in the number of clusters for each singular height
###
###         ** 'POISONf' -- the required probability of poisoning activity, given as an
###                     integer in [0,100]
###
###         ** 'TRUNC' -- a list of truncation heights in [0,1], relative to the diameter of 
###                     of the data set
###
###         ** 'DETECTION_THRESHOLD' -- detection threshold in [0,1] for the entropy-difference measure:
###             - if $abs(1-DH(new_dendrogram)/DH(old_dendrogram))<=threshold$ then no detection.
###
###         ** 'PERFORMANCE' -- a list of [keys to] performance rates to be plotted.
###             - each key must be mentioned & defined in hec_plot.py
###
###         ** 'MEASURES' -- a list of [keys to] measures to be reported at each step.
###             - the measures must be defined in slhc_GPU_wrpper.py.
###

### OUTPUT:
### A run of the program will create two sub-directories, named 'active' and 'passive' inside
### the run subdirectory "./$dir_name$/". Each will contain a collection of files named 
### 'run_Tn_Rx.out', each containing a dictionary with data about that run:
###
###     * $n in xrange(len(PREAMBLE('TRUNC')))$, and $x in xrange(PREAMBLE('RUNS'))$;
###
###     * all the following dictionary items are lists, with one element per each step of the run:
###
###     * 'samples_submitted':      the sample (feature vec) that was submitted to the DB
###
###     * 'poisonQ' values:         1 if the sample was a poison pill, 0 otherwise;
###
###     * 'acceptedQ':              1 to accept, 0 to reject;
### 
###     * 'detectableQ':            0 sample is too close to any cluster to be declared anomalous.
###
###     * an additional dictionary entry, 'measures', is itself a dictionary:
###
###         ** for each requested $meas$ in $PREAMBLE['MEASURES']$, a list of values of the
###             requested measures will be provided, one value per each step of the run.
###

#def one_run(run_type,data_set,rperm,poison_sequence,preamble,counter_trunc,counter_run):
def one_run(preamble,scn,run_type,counter_trunc,counter_run):
    #prepare output file
    try:
        out_file=open(preamble[run_type]+'_T'+str(counter_trunc)+'_R'+str(counter_run)+'.out','wb')
    except KeyError:
        raise Exception('Unsupported type of run -- Aborting!\n')

    pickler=cPickle.Pickler(out_file,protocol=cPickle.HIGHEST_PROTOCOL)

    #prepare DB for this run
    RUN_PARAMS={'MODE':run_type,'TRUNC'=counter_trunc,'RUNNO'=counter_run}
    DB=poison.DHCDB_classes[preamble['POISONm']](preamble,scn,RUN_PARAMS)

    total_samples=len(DB._data)+len(DB._extra)
    cycle_counter=len(DB._data)

    #output is recorded here
    out_dict={
        'samples_submitted':[],
        'poisonQ':[],
        'acceptedQ':[],
         'measures':{meas:[] for meas in preamble['MEASURES']},
        }

    #while target number of cycles is not reached, continue the current run
    while cycle_counter<=preamble['SAMPLE_FINAL']*total_samples:
        # add a sample to the database
        new_sample,poisonQ,acceptedQ,detectableQ,measures=DB.new_sample(poison_sequence[cycle_counter])
        # append output to results:
        out_dict['samples_submitted'].append(new_sample)
        out_dict['poisonQ'].append(poisonQ)
        out_dict['acceptedQ'].append(acceptedQ)
        for meas in preamble['MEASURES']:
            out_dict['measures'][meas].append(measures[meas])
        cycle_counter+=1
    pickler.dump(out_dict)

    #close output file and wrap up
    out_file.close()
    #return None

def one_run_unpack(args):
    return one_run(*args)

def main():
    my_pool=Pool(processes=N_CORES)

    # obtain data & directory names
    dir_name=sys.argv[1]

    if sys.platform=='linux2':
        path_name=sys.path[0]+'/'+dir_name+'/'
        path_name_active=path_name+'active/'
        path_name_passive=path_name+'passive/'
    else:
        path_name=sys.path[0]+'\\'+dir_name+'\\'
        path_name_active=path_name+'active\\'
        path_name_passive=path_name+'passive\\'
    try:
        #mkdir(path_name)
        mkdir(path_name_active)
        mkdir(path_name_passive)
    except:
        pass
        #raise Exception('This directory exists already -- ABORTING!')

    #load the preamble and the data file
    try:
        #load the preamble file:
        PREf=open(path_name+dir_name+'.pre','rb')
        PREAMBLE=cPickle.load(PREf)
        PREf.close()
        #augment the preamble:
        PREAMBLE['active']=path_name_active
        PREAMBLE['passive']=path_name_passive
    except:
        raise Exception('Something is wrong with your data files -- ABORTING!\n')
    
    
    try:
        #load the data file
        DATAf=open(path_name+dir_name+'.dat','rb')
        DATA=[np.array(item) for item in pickle_loader(DATAf)]
        DATAf.close()
    except:
        raise Exception('Something went wrong reading the data file ['+path_name+dir_name+'.dat] -- ABROTING!\n')

    #read data into scenario
    SCN=scenario()
    SCN.get_data(DATA)

    #execute the runs
    for N in xrange(PREAMBLE['RUNS']):
        message='Starting run number '+str(N+1)+' of '+str(PREAMBLE['RUNS'])
        print '\n'+message+'\n'+'-'.join(['' for ind in xrange(len(message))])

        #generate a scenario
        SCN.make_permutation()
        SCN.make_poisoner(PREAMBLE['POISONf'])
        
        #generate the corresponding runs with the different parameters
        for RUN_TYPE in ['passive','active']:
            print '\n - '+RUN_TYPE+' sequence ('+str(len(PREAMBLE['TRUNC']))+' cycles): '
            N_truncs=len(PREAMBLE['TRUNC'])
            # construct iterator over the different truncation values
            #my_it=[(RUN_TYPE,DATA,rand_perm,poisonQ,PREAMBLE,K,N) for K in xrange(len(PREAMBLE['TRUNC']))]
            my_it=[(PREAMBLE,SCN,RUN_TYPE,K,N) for K in xrange(len(PREAMBLE['TRUNC']))]
            my_pool.map(one_run_unpack,my_it)
            #map(one_run_unpack,my_it)
    
    #close the pool when done
    my_pool.close()
    my_pool.join()


if __name__=='__main__':
    main()

