### module for basic operations with ellone metric on Boolean vectors
from scipy.spatial.distance import cityblock as ellone
from scipy.spatial.distance import pdist
from numpy.random import randint as rnd
from numpy.random import permutation as npperm
import numpy as np

def get_metric(params=None):
    met_func=lambda pt1,pt2: ellone(pt1,pt2) if params is None else lambda pt1,pt2: ellone(params*pt1,params*pt2)
    matr_func=lambda points_list: pdist(points_list,metric='cityblock') if params is None else lambda points_list: pdist([params*pt for pt in points_list],metric='cityblock')
    return met_func,matr_func

### diff of two Boolean vectors (even if given with 1.0/0.0 instead of True/False)
diff=np.vectorize(lambda x,y: 1 if (0.25<=0.5*(x+y)<=0.75) else 0)

### (mod 2) sum of two Boolean vectors (even if provided as floats)
nearone=np.vectorize(lambda x: 1 if 0.75<=x<=1.25 else 0)
def boolsum(x,y): #assume x,y are np.arrays of the same shape...
    return nearone(x+y)

### RANDOM SUBMASK OF SIZE HALF
def choose_half(mask): #ASSUMPTION: mask is a numpy array
    indices=np.compress(mask,range(len(mask)))
    l=len(indices)
    indices=npperm(indices)[:l/2]
    np.putmask(mask,[(ind in indices) for ind in xrange(len(mask))],np.zeros_like(mask))

### Compute a midpoint
def midpoint(pt1,pt2):
    mask=boolsum(pt1,pt2)
    choose_half(mask)
    return boolsum(pt1,mask)

def main():
    x=np.array([1,1,0,0,0,0,1,1])
    y=np.array([1,0,1,0,1,0,1,0])
    print '\nX = '+str(x)
    print 'Y = '+str(y)
    print '\n Some choices of midpoints:'
    print 'M = '+str(midpoint(x,y))
    print 'M = '+str(midpoint(x,y))
    print 'M = '+str(midpoint(x,y))
    exit(0)

if __name__=="__main__":
    main()