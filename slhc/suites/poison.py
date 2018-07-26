#IO
import sys
from scipy.io import loadmat
import cPickle


#General Python
from copy import deepcopy as cp
#from collections import deque


#Graphics
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#Computational
import numpy as np
from numpy.random import randint as rnd
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform


#Specific
import slhc_GPU_wrapper as slhc


#scenario-dependent initialization for the data set.
def split_data_set(preamble,scn):
    N=len(scn.data())
    data=[] # prepare container for data base content
    extra=[] # prepare container for `environment' samples
    for count in xrange(N):
        sample=scn.data(count)
        if count<preamble['SAMPLE_INITIAL']*N: #obtain the required fraction of random samples from the data set
            data.append(sample)
        else: #set the remaining samples aside
            extra.append(sample)
    return data,extra



#Setting up a GPU-enabled SLHC database of Euclidean points with minimal
#assimilation, poisoning and deliberation functionalities:
#-  assimilation: add the point to the library
#-  deliberation: DH-based accept/reject
#-  poisoning: just insert a "regular" sample
#
class poison_DB_Base_Class():
    def __init__(self,preamble,scn,run_params):
        ##prepare scn attributes
        self._scn=scn
        self._preamble=preamble
        self._clock=0

        ##prepare run attributes
        self._mode=run_params['MODE']

        ##read initial data set
        self._data,self._extra=split_data_set(preamble,scn)
        self._initial_size=len(self._data)

        ##prepare metric and slhc object
        if preamble['METRIC']=='elltwo':
            import module_elltwo as metric
        elif preamble['METRIC']=='elltwo_normalized':
            import module_elltwo_norm as metric
        elif preamble['METRIC']=='ellone':
            import module_ellone as metric
        elif preamble['METRIC']=='ellone_Boolean':
            import module_ellone_bool as metric
        else:
            raise Exception('Input metric \"'+preamble['METRIC']+'\" unavailable. -- ABORTING!\n')
        self._metric,self._dmatrix=metric.get_metric() #no parameters/stretching for now
        self._midpoint=metric.midpoint

        ##prepare slhc objects:
        self._slhc_DB=slhc.GPU_initialize_slhc(slhc.u2l(self._dmatrix(self._data)))
        self._slhc_WORK=slhc.GPU_initialize_slhc(slhc.u2l(self._dmatrix(self._data)))
        self._slhc_DB.calculate("slhc")
        self._slhc_WORK.calculate("slhc")
        
        ##truncation height for this run equals...
        ##...assigned fraction of the height of the initial hierarchy
        ## NOTE: run_params['TRUNC'] holds the ORDINAL NUMBER of the actual truncation
        ##       height [ratio] listed in preamble['TRUNC'].
        self._trunc=preamble['TRUNC'][run_params['TRUNC']]*self._slhc_DB.report("height")[0]
        #print list(self._slhc_DB.report("height"))
        #print preamble['TRUNC'][run_params['TRUNC']]
        #print self._trunc
        
    def get_distances(self,sample):
        return np.array([self._metric(sample,pt) for pt in self._data])
    
    # Obtain and process a new sample
    def new_sample(self):
        #generate the sample
        poisonedQ=self.poisoner()
        new_pt=self.poison_pill() if poisonedQ else self.honest_sample()
        
        #expand the work object, compute ultra-metric and restrict
        self._slhc_WORK.insert(self.get_distances(new_pt))
        self._slhc_WORK.calculate("slhc")
        self._slhc_WORK.remove_one()

        #deliberation
        acceptedQ=self.deliberate()

        #collecting measures
        measures={}
        for key in self._preamble['MEASURES']:
            try:
                measures[key]=slhc.DISTS[key](self._slhc_DB,self._slhc_WORK,self._trunc)
            except KeyError:
                raise Exception('Attempted computing unavailable measure \$'+key+'\$ -- Aborting!\n')
        
        if self._mode=='passive':
            self.assimilate(new_pt)
        elif self._mode=='active':
            if acceptedQ: #assimilate the sample if accepted
                self.assimilate(new_pt)
            else: 
                self.reject(new_pt)
        else:
            raise Exception('\nMode not implemented -- ABORTING!\n')

        self._clock+=1
        return new_pt,poisonedQ,acceptedQ,measures

    def poisoner(self): #default poisoner is based on the supplied scenario
        return self._scn.poison(self._initial_size+self._clock)
         
    def poison_pill(self): # poisoning: poisoner samples from the ground truth
        return self._extra.pop()

    def honest_sample(self):
        return self._extra.pop()
    
    def deliberate(self): # standard DH-based deliberation
        #compute maximum possible entropy for dendrogram this size
        maxent=slhc.lg(len(self._data))
        #accept if entropy difference does not overshoot threshold*maxent
        return abs(self._slhc_DB.entropy(self._trunc)-self._slhc_WORK.entropy(self._trunc))<=self._preamble['DETECTION_THRESHOLD']*maxent

    #STOPPED HERE
    def assimilate(self,new_pt): # assimilation: add the sample to the database
        # add the new point to the slhc object
        new_dists=self.get_distances(new_pt)
        self._slhc_DB.insert(new_dists)
        self._slhc_DB.calculate("slhc")
        # add the new point to the work object
        self._slhc_WORK.insert(new_dists)
        self._slhc_WORK.calculate("slhc")
        # add the point to the data set
        self._data.append(new_pt)
        return None
        
    def reject(self,new_pt):
        return None

    ### MISC METHODS REQUIRED BY SEVERAL SUBCLASSES...
    def bridges(self):
        #form a minimum spanning tree        
        mst=minimum_spanning_tree(csr_matrix(squareform(self._dmatrix(self._data)))).todok().keys()

        #form and return the list of midpoints of edges in the MST
        bridge_list=[]
        for i,j in mst:
            bridge_list.append((
                self._midpoint(self._data[i],self._data[j]),
                self._metric(self._data[i],self._data[j])
                ))
        return bridge_list


### BridgeBest class implementation
class poison_best_DB(poison_DB_Base_Class):
    def poison_pill(self): # generates a BridgeBest poison pill
        #obtain notion of distance between clusterings
        dist=slhc.DISTS[self._preamble['POISONd']]
        #compute all bridge_points,bridge_lengths on the data base
        bridges=self.bridges()

        #rank the bridge points as poison pills
        rankings=[]
        for bridge_point,bridge_length in bridges:
            self._slhc_WORK.insert(self.get_distances(bridge_point))
            self._slhc_WORK.calculate("slhc")
            self._slhc_WORK.remove_one()
            rankings.append(dist(self._slhc_DB,self._slhc_WORK,0.5*bridge_length))
        pt,_=bridges[np.argmax(rankings)]
        return pt


### BridgeRandom class implementation
class poison_random_DB(poison_DB_Base_Class):
    def poison_pill(self): # generate a random "bridging" poison pill
        bridges=self.bridges()
        pt,_=bridges[rnd(len(bridges))]
        return pt
            
DHCDB_classes={
    'BaseClass':poison_DB_Base_Class,
    'BridgeBest':poison_best_DB,
    'BridgeRandom':poison_random_DB,
    }


def main():
    import module_scenario
    # Load the data file
    SOURCE='./hierarch_small.dat'
    SCN=module_scenario.scenario()
    SCN.load_data(SOURCE)
    SCN.make_permutation()
    SCN.make_poisoner()

    # Construct preamble
    PREAMBLE={}
    PREAMBLE['RUNS']=1
    PREAMBLE['SAMPLE_INITIAL']=0.5
    PREAMBLE['SAMPLE_FINAL']=0.75

    PREAMBLE['METRIC']='elltwo'
    PREAMBLE['MEASURES']=['deltaDH']

    PREAMBLE['POISONm']='BridgeBest' #another option 'BridgeRandom'
    PREAMBLE['POISONd']='Biggio'
    PREAMBLE['POISONf']=50

    PREAMBLE['TRUNC']=[pow(0.5,k+1) for k in xrange(4)]
    PREAMBLE['DETECTION_THRESHOLD']=0

    PREAMBLE['active']='.\\poison_test\\active\\' if sys.platform=='win32' else './poison_test/active/'
    PREAMBLE['passive']='.\\poison_test\\passive\\' if sys.platform=='win32' else './poison_test/passive/'

    #run parameters
    RUN_PARAMS={'MODE':'active','TRUNC':2}

    #Total number of steps for test
    STEPS=20
    fig,ax=plt.subplots()

    #GLOBALS
    
    DB=DHCDB_classes[PREAMBLE['POISONm']](PREAMBLE,SCN,RUN_PARAMS)
    N0=len(DB._data)
    x_coords={}
    y_coords={}
    x_coords['go']=[pt[0] for pt in DB._data]
    y_coords['go']=[pt[1] for pt in DB._data]
    for key in ['bs','rs','bx','rx']:
        x_coords[key]=[]
        y_coords[key]=[]
    markers=['go','bs','rs','bx','rx']
    print "Green dot = initial datum"
    print "Square/X  = accepted / rejected"
    print "Red/Blue  = poison pill / true sample"
    ax.grid()
    ax.set_ylim(-0.1,1.1)
    ax.set_xlim(-0.1,1.1)
    scatter_dict={key:ax.scatter(x_coords[key],y_coords[key],color=key[0],marker=key[1]) for key in markers}
 
    #data initialization for plot
    def init_data():
        return scatter_dict
        #return scatter_dict['go'],scatter_dict['bs'],scatter_dict['rs'],scatter_dict['bx'],scatter_dict['rx'],

    #data generation for plot
    def generate_data(steps):
        counter=0
        while DB._extra and counter<steps:
            counter+=1
            #print DB._slhc.report('dist')
            yield DB.new_sample()

    #One poisoning step
    def animation_step(data):
        new_pt,poisonedQ,acceptedQ,_=data
        for key in markers:
            if key!='go' and key==('r' if poisonedQ else 'b')+('s' if acceptedQ else 'x'):
                x_coords[key].append(new_pt[0])
                y_coords[key].append(new_pt[1])
                scatter_dict[key].set_offsets(zip(x_coords[key],y_coords[key]))
        return scatter_dict
        #return scatter_dict['go'],scatter_dict['bs'],scatter_dict['rs'],scatter_dict['bx'],scatter_dict['rx'],

    #run the test animation
    ani=animation.FuncAnimation(fig,animation_step,frames=generate_data(STEPS),init_func=init_data,interval=500)
    #print 'Done!\n'
    plt.show()
        
if __name__=="__main__":
    main()
    
