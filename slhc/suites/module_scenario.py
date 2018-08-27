### SIMULATION SCENARIO OBJECT
import cPickle
from numpy.random import randint as rnd
from numpy.random import permutation as npperm
import numpy as np

def pickle_loader(pickled_file):
    try:
        while True:
            yield cPickle.load(pickled_file)
    except EOFError:
        pass

def get_points(filename):
    myf=open(str(filename),'rb')
    tmp_list=[np.array(item) for item in pickle_loader(myf)]
    myf.close()
    return np.array(tmp_list)

class scenario():
    def __init__(self):
        self._data=np.array([])
        self._perm=[]
        self._poison=[]

    #load/generate content
    def load_data(self,filename):
        self._data=get_points(filename)

    def get_data(self,data): #assuming $data$ is a list/tuple
        self._data=np.array(data)

    def get_permutation(self,perm):
        if len(perm)<=len(self._data):
            self._perm=perm
        else:
            raise Exception('\nPermutation support larger than data vector -- ABORTING!\n')

    def get_poisoner(self,poison):
        if len(poison)<=len(self._data):
            self._poison=poison
        else:
            raise Exception('\nPoison vector longer than data vector -- ABORTING!\n')
        
    def make_permutation(self):
        self._perm=npperm(len(self._data))

    def make_poisoner(self,percentage=50): # $percentage$ needs to be an integer from 0 to 100
        self._poison=[rnd(100)<percentage for ind in xrange(len(self._data))]

    
    #report content:
    def data(self,ind=None):
        if ind is None:
            return self._data
        else:
            return self._data[self._perm[ind]]

    def perm(self,ind=None):
        if ind is None:
            return self._perm
        else:
            return self._perm[ind]

    def poison(self,ind=None):
        if ind is None:
            return self._poison
        else:
            return self._poison[ind]

  
