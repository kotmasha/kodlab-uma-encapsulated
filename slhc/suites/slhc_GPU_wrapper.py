import numpy as np
import dslhc

from collections import deque
from scipy.sparse import dok_matrix


### base-two logarithms
def lg(x):
    return np.log(x)/np.log(2)

### SLHC IMPLEMENTED ON GPU WRAPPER

class slhc_GPU_wrapper:
    def __init__(self):
        self.__world__ = dslhc.World()
        self.__point_num__ = 0
        self.__pair_num__ = 0
        #self.__dendrogram__= {} #dictionary of form cluster:height,children

    """
    BASIC FUNCTIONALITY
    """
    def insert(self, dist):
        """
        :param dist: the input dist numpy array, input size should be the same as the current point size. 0 will be added in the end
        :return:
        """
        row = dist.tolist()
        if self.__point_num__ is not len(dist):
            raise Exception("Input distance not correct, should input %d, but receive %d" %(self.__point_num__, len(dist)))
        row.append(0)
        self.__world__.insert(row)
        self.__point_num__ += 1
        self.__pair_num__ += self.__point_num__

    def remove_rows(self, idxs):
        """
        :param idxs: the idxs list to remove
        :return:
        """
        m1 = max(idxs)
        m2 = min(idxs)
        if m2 < 0 or m1 >= self.__point_num__:
            raise Exception("The input idxs range is not correct, detected the range is %d-%d, should be 0-%d" %(m2, m1, self.__point_num__ - 1))
        self.__world__.remove_rows(idxs)
        self.__point_num__ -= len(idxs)
        self.__pair_num__ = self.__point_num__ * (1 + self.__point_num__) / 2

    def remove_one(self):
        self.remove_rows([self.__point_num__ - 1])

    def size(self):
        return self.__point_num__

    def report(self, key):
        if key is not "slhc" and key is not "height" and key is  not "dist":
            raise Exception("The input key is not supported")
        elif key=="dist" or key=="slhc":
            mat=list(self.__world__.report(key))
            tmp=[]
            for n in range(self.__point_num__,0,-1):
                tmp.append(mat[-n:-1])
                mat=mat[:-n]
            mat=[]
            while tmp:
                mat+=tmp.pop()
            return mat
        else:
            return self.__world__.report(key)

    """
    COMPUTATIONAL, GPU-SIDE
    """
    def calculate(self, key):
        if key is "slhc":
            #compute the ultra-metric
            self.__world__.calculate(key)
        else:
            raise Exception("No key other than slhc is supported now!")

    def blocks(self, delta):
        return self.__world__.blocks(delta)


    """
    COMPUTATIONAL, PYTHON SIDE
    """
    # Discrete entropy of a dendrogram, realized as
    # \frac{1}{|X|}*\left(
    #     (|X|-1)log|X|+
    #     \sum_{heights\geq trunc}\sum_{blocks}(1-|ch(B)|)log|B|
    # \right)
    def entropy(self,trunc=0):
        heights=[item for item in self.report("height") if item>trunc]+[trunc]
        ent=(self.size()-1)*lg(self.size())

        fine_blocks=[frozenset(['dummy'])]
        while heights:
            coarse_blocks=[frozenset(item) for item in self.blocks(heights.pop())]
            m=1-sum(np.array([[b<=c for c in coarse_blocks] for b in fine_blocks]))
            n=np.array([lg(len(c)) for c in coarse_blocks])
            ent+=sum(m*n)
            fine_blocks=coarse_blocks
        return (1./self.size())*ent

    # partition incidence matrix (in dok_matrix sparse format)
    # for the given height
    def HeightIncidence(self,height):
        clusters=[list(item) for item in self.blocks(height)]
        m=dok_matrix((self.size(),len(clusters)),dtype=int)
        for x in xrange(self.size()):
            for u in clusters:
                m[x,u]=int(x in u)
        return m
    # adjacency matrix for the given height, in dok_matrix sparse format
    def HeightAdjacency(self,height):
        m=self.HeightIncidence(height)
        return m*m.transpose()

    def dist_HeightAdjacency(self,other,height):
        m=self.ClusterAdjacency(height)
        n=other.ClusterAdjacency(height)
        return sum(sum(np.abs(np.array((m-n).todense()))))        

    # cluster incidence matrix for all clusters
    def ClusterIncidence(self,truncation_height=0):
        clusters=set([])
        heights=[h for h in list(self.report("height")) if h>truncation_height]+[truncation_height]
        for h in heights:
            #print clusters
            new_clusters={frozenset(item) for item in self.blocks(h)}
            clusters.update(new_clusters)
        m=dok_matrix((self.size(),len(clusters)),dtype=int)
        for x in xrange(self.size()):
            for ind,u in enumerate(clusters):
                m[x,ind]=x in u
        return m


    # cluster adjacency matrix in dok_matrix sparse format
    def ClusterAdjacency(self,truncation_height=0):
        m=self.ClusterIncidence(truncation_height)
        return m*m.transpose()
    

    # cluster adjacency distance
    def dist_ClusterAdjacency(self,other,truncation_height=0):
        m=self.ClusterAdjacency(truncation_height)
        n=other.ClusterAdjacency(truncation_height)
        return sum(sum(np.abs(np.array((m-n).todense()))))

    
    
### CONDENSED ROW FORMAT FOR DISTANCE MATRICES: LOWER-TO-UPPER AND BACK
def l2u(arr):
    n=(1+int(pow(1+8*len(arr),0.5)))/2
    return np.array([arr[i+j*(j-1)/2] for i in xrange(n) for j in xrange(i+1,n)])

def u2l(arr):
    n=(1+int(pow(1+8*len(arr),0.5)))/2
    return np.array([arr[(n-1)*j-j*(j+1)/2+i-1] for i in xrange(1,n) for j in xrange(i)])


### STANDARD INITIALIZER FOR SLHC OBJECT FROM A DISTANCE MATRIX
def GPU_initialize_slhc(ltrc_matrix):
    slhc_obj=slhc_GPU_wrapper()
    ind=0
    end=False
    while not end:
        try:
            first=((ind-1)*ind)//2
            last=(ind*(ind+1))//2
            slhc_obj.insert(ltrc_matrix[first:last])
            ind+=1
        except:
            end=True
    slhc_obj.calculate('slhc')
    return slhc_obj

### dissimilarity measures between hierarchies

#dissimilarity based on adjacency matrices
def dAC(slhc1,slhc2,truncation_height=0): #assumes both objects have up-to-date hierarchies
    return slhc1.dist_ClusterAdjacency(slhc2,truncation_height)

#dissimilarity based on individual levels (and their adjacency matrices)
def dWH(slhc1,slhc2,truncation_height=0): #assumes both objects have up-to-date hierarchies
    fil=lambda x: x>truncation_height
    sl_heights1=filter(fil,list(slhc1.report("height")))
    sl_heights2=filter(fil,list(slhc2.report("height")))
    #print sl_heights1
    #print sl_heights2
    heights_list=[truncation_height]+sl_heights1+sl_heights2
    return max([slhc1.dist_HeightAdjacency(slhc2,height) for height in heights_list])

# "worst cut" dissimilarity considered by Biggio and his people?
def dBiggio(slhc1,slhc2,truncation_height=0): #assumes both objects have up-to-date hierarchies
    sl_heights1=list(slhc1.report("height"))
    sl_heights2=list(slhc2.report("height"))
    max_height=max(sl_heights1[0],sl_heights2[0])
    h1=[h for h in sl_heights1 if truncation_height<=h<=max_height]
    h2=[h for h in sl_heights2 if truncation_height<=h<=max_height]
    heights_list=list(set(h1)|set(h2))
    return min([slhc1.dist_HeightAdjacency(slhc2,height) for height in heights_list])

def dDH(slhc1,slhc2,truncation_height=0):
    return slhc1.entropy(truncation_height)-slhc2.entropy(truncation_height)

DISTS={
    'AdjacencyAllClusters':dAC,
    'AdjacencyWorstHeight':dWH,
    'Biggio':dBiggio,
    'deltaDH':dDH,
}

### Test script:
def main():
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    PTS=np.array([[0,0],[1,1],[3,2],[3,3],[2,4],[1,0]])

    print '\nGiven the list of points in the plane:'
    print PTS

    print '\nLoad the distance matrix into an slhc object...'
    DMATRIX=pdist(PTS,'cityblock')
    print '\t'+str(DMATRIX)
    print '\n\tTransforming matrix into lower-triangular compressed form:'
    print '\t'+str(u2l(DMATRIX))
    SLOB=GPU_initialize_slhc(u2l(DMATRIX))

    print '\nDistance:'
    print squareform(l2u(np.array(SLOB.report("dist"))))
    print '\nSLHC distance:'
    print squareform(l2u(np.array(SLOB.report("slhc"))))

    print '\nForm the dendrogram from the current slhc contents:'
    for height in list(SLOB.report("height")):
        print str([list(item) for item in SLOB.blocks(height)])+'\tat height '+str(height)

    print '\nResulting in the following cluster incidence and adjacency matrices:'
    print SLOB.ClusterIncidence().todense()
    print SLOB.ClusterAdjacency().todense()

    print '\nAnd having entropy '+str(SLOB.entropy())+' out of '+str(lg(SLOB.size()))+' bits.'

def main2():
    import cPickle
    import sys
    from scipy.spatial.distance import euclidean
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage
    
    FILENAME='hierarch_small.dat'
    #FILENAME='banana_small.dat'
    try:
        NOH=int(sys.argv[1])
    except:
        NOH=5
    
    def points(filename):
        with open(filename,'rb') as myf:
            while myf:
                try:
                    yield cPickle.load(myf)
                except:
                    break

    DATA=[]
    SLOB=slhc_GPU_wrapper()
    for npt in points(FILENAME):
        #print npt
        SLOB.insert(np.array([euclidean(npt,pt) for pt in DATA]))
        DATA.append(npt)

    SLOB.calculate("slhc")
    SLOB_heights=np.array(SLOB.report("height"))

    print "\nTop "+str(NOH)+" heights computed by dslhc (Siqi):"
    print SLOB_heights[:NOH]

    LINK=linkage(DATA)
    #print LINK
    LINK_heights=np.sort(np.array(list(set([LINK[i][2] for i in xrange(len(LINK))]))))[::-1]

    print "\nTop 5 heights computed by linkage (SciPy):"
    print LINK_heights[:NOH]

if __name__=='__main__':
    main2()
