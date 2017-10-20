import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cityblock as ellone
from scipy.spatial.distance import euclidean as elltwo
from scipy.cluster.hierarchy import linkage
from numpy import log as lg
from numpy.random import randint as rnd
from numpy.random import permutation as npperm
from copy import deepcopy as cp
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

### diff of two Boolean vectors
diff=np.vectorize(lambda x,y: 1 if (0.25<=0.5*(x+y)<=0.75) else 0)

### boolsum of two Boolean vectors
nearone=np.vectorize(lambda x: 1 if 0.75<=x<=1.25 else 0)
def boolsum(x,y): #assume x,y are np.arrays of the same shape...
    return nearone(x+y)

### CONDENSED FORMAT DISTANCE MATRIX LOWER-TO-UPPER AND BACK
def l2u(arr):
    n=(1+int(pow(1+8*len(arr),0.5)))/2
    return np.array([arr[i+j*(j-1)/2] for i in xrange(n) for j in xrange(i+1,n)])

def u2l(arr):
    n=(1+int(pow(1+8*len(arr),0.5)))/2
    return np.array([arr[(n-1)*j-j*(j+1)/2+i-1] for i in xrange(1,n) for j in xrange(i)])


### MATRIX [TENSOR] DOUBLE
def double(A):
    return A*np.transpose(A)

### ELEMENTWISE 1-NORM OF A MATRIX
def norm1(A):
    #ASSUME $A$ IS A NUMPY MATRIX
    return sum(sum(np.array(np.abs(A))))

### RANDOM SUBMASK OF GIVEN SIZE
def choose_half(mask): #ASSUMPTION: mask is a numpy array
    indices=np.compress(mask,range(len(mask)))
    l=len(indices)
    indices=npperm(indices)[:l/2]
    np.putmask(mask,[(ind in indices) for ind in xrange(len(mask))],np.zeros_like(mask))

### BINARY SORTING

def merge(lis1,lis2):
    merged=[]
    ind1=0
    ind2=0
    while ind1<len(lis1) and ind2<len(lis2):
        if lis1[ind1]<=lis2[ind2]:
            merged.append(lis1[ind1])
            ind1+=1
        else:
            merged.append(lis2[ind2])
            ind2+=1
    merged.extend(lis1[ind1:])
    merged.extend(lis2[ind2:])
    return merged

def binsort(lis):
    if len(lis)<=1:
        return lis
    else:
        head=binsort([item for ind,item in enumerate(lis) if ind<len(lis)/2])
        tail=binsort([item for ind,item in enumerate(lis) if ind>=len(lis)/2])
        return merge(head,tail)


### OBJECT CLASS FOR DENDROGRAMS ("HIERARCHIES")
class hnode(object):
    def __init__(self,underlying_set,height,parent=None):
        self._underlying=set(underlying_set)
        self._parent=parent
        self._children=[]
        self._height=height
        self._combed=False

    def __repr__(self): #,count=0):
        rep=str(self._underlying)+' at height '+str(self._height)+'\n'
        rep+='*\t'.join([child.__repr__() for child in self._children])
        #for child in self._children:
        #    rep+=times('*\t',count+1)+str(child.__repr__(count+1))
        return rep

    # Return the cardinality of the underlying set
    def card(self):
        return len(self._underlying)
        
    # Adding an existing hnode as a child to $self$
    def add_child(self,child_node):
        self._height=max([self._height,child_node._height])
        self._underlying.update(child_node._underlying)
        self._children.append(child_node)
        child_node.parent=self
        self._combed=False

    # Who is my root?
    def root(self):
        if self._parent!=None:
            return self._parent.root()
        else:
            return self

    # Am I a leaf of the dendrogram?
    def is_leaf(self):
        return not self._children

    # Comb the dendrogram heredown
    def comb(self):
        if self._combed:
            exit
        new_children=[] #prepare empty list to be populated by combed children
        for child in self._children: #iterate over children
            child.comb() #comb the current child
            if child.is_leaf() or child._height<self._height:
                if len(child._underlying)<len(self._underlying):
                    #keep child if its content is shorter
                    new_children.append(child)
                else:
                    #remove child, 
                    self._height=child._height
                    self._children=[]
            else:
                new_children.extend(child._children)

        self._children=new_children
        self._combed=True
    
    # Slice at height $height$ (outputs a partition)
    def slice(self,height):
        if not self._combed:
            self.comb()
        else:
            pass
        if height>=self._height:
            return [binsort(list(self._underlying))]
        else:
            slice=[]
            for child in self._children:
               slice.extend(child.slice(height)) 
            return binsort(slice)

    # Slice at height $height$ (outputs incidence matrix)
    def slice_incidence_matrix(self,height):
        return np.matrix([[int(y in x) for x in self.slice(height)] for y in self._underlying])

    # Outputs clusters heredown
    def clusters(self):
        if not self._combed:
            self.comb()
        else:
            pass
        cluster_list=[list(self._underlying)]
        for child in self._children:
            cluster_list.extend(child.clusters())
        return binsort(cluster_list)
        
    # Outputs incidence matrix of clusters heredown
    def clusters_incidence_matrix(self):
        return np.matrix([[int(y in x) for x in self.clusters()] for y in self._underlying])
    
    # Truncate self at height $delta$
    def truncate_self(self,delta):
        if not self._combed:
            self.comb()
        else:
            pass
        if self._height<=delta:
            self._height=0
            self._children=[]
        else:
            for child in self._children:
                child.truncate_self(delta)

    # Return a truncated copy of self (at height $delta$)
    def truncated(self,delta):
        new_tree=cp(self)
        if not new_tree._combed:
            new_tree.comb()
        else:
            pass
        new_tree.truncate_self(delta)
        return new_tree

    # Restrict self to a subset of the underlying set
    def restrict_self(self,new_underlying,parent=None):
        status=self._underlying.intersection(new_underlying)
        if status: #if intersection non-empty, update and go to children
            self._underlying.intersection_update(new_underlying)
            for child in self._children:
                child.restrict_self(new_underlying,self)
        else: #if intersection is emtpy, remake parent into leaf
            parent._children.remove(self)
            if not parent._children:
                parent._height=0
        if not parent:
            self.comb()

    # Return a restricted copy of self
    def restricted(self,new_underlying):
        new_tree=cp(self)
        if not new_tree._combed:
            new_tree.comb()
        else:
            pass
        new_tree.restrict_self(new_underlying)
        return new_tree

    # Return an ascending list of branching heights
    def heights(self):
        heights_list=[self._height]
        for child in self._children:
            heights_list.extend(child.heights())
        return binsort(list(set(heights_list)))
        
    # What is my entropy?
    def entropy(self):
        return (1./self.card()) * sum( (child.card() * child.entropy() - (child.card()-1.) * (lg(child.card())-lg(self.card()))) for child in self._children )

### EXTRACTING A DENDROGRAM STRUCTURE FROM A LINKAGE MATRIX:
### INPUT:
### Assuming $link$ is a (n-1)-by-4 matrix such that:
### - $n$ is the number of particles
### - clusters indexed link[i][0] and link[i][1] are combined into cluster n+i
### - indices 0,...,n-1 correspond to leaves (original observations)
### - link[i][2] is the linkage value of cluster n+i
### - link[i][3] is the cardinality of cluster n+i
###
### OUTPUT:
### A list of partitions of the original observations, starting with resolution
### zero.
def linkage_to_dendrogram(link):
    n=len(link)+1

    new_link=[[[ind],0.,1.] for ind in xrange(n)]
    new_link.extend([[[np.int_(link[ind,0]),np.int_(link[ind,1])],link[ind,2],link[ind,3]] for ind in xrange(n-1)])

    def generate_node(ind):
        if ind in xrange(n):
            return hnode([ind],0.)
        else:
            new_node=hnode([],new_link[ind][1])
            for indc in new_link[ind][0]:
                new_node.add_child(generate_node(indc))
            return new_node

    return generate_node(2*n-2)

### distances between dendrograms

## Discrepancy between cluster-based adjacency matrices 
def dCA(root1,root2,extra=None):
    # The input $extra$ is a dummy, introduced to facilitate calls to other discrepancy measures requiring extra input, such as $dWC$
    if root1._underlying!=root2._underlying:
        raise Exception('Underlying sets must match to compute the distance between two hierarchies.')
    else:
        return norm1(double(root1.clusters_incidence_matrix())-double(root2.clusters_incidence_matrix()))

## Biggio's "worst-cut" dissimilarity
def dWC(root1,root2,min_height):
    # The input $min_height$ is the lower bound on the heights considered for minimizations 
    if root1._underlying!=root2._underlying:
        raise Exception('Underlying sets must match to compute the distance between two hierarchies.')
    else:
        cut_distances=[]
        max_height=min(max(root1.heights()),max(root2.heights()))
        #print '\nHeight limits: min '+str(min_height)+', max '+str(max_height)+'\n'
        h1=[h for h in root1.heights() if h>=min_height and h<=max_height]
        h2=[h for h in root2.heights() if h>=min_height and h<=max_height]
        heights_list=list(set(merge(h1,h2)))
        #print '\nOriginal heights:'
        #print h1
        #print '\nNew heights:'
        #print h2
        #print '\n'
        for height in heights_list:
            cut_distances.append(
                norm1(double(root1.slice_incidence_matrix(height))-double(root2.slice_incidence_matrix(height)))
                )

        return min(cut_distances)

## Difference in hierarchical entropy 
def dDH(root1,root2,truncation_height=0):
    if root1._underlying!=root2._underlying:
        raise Exception('Underlying sets must match to compute the distance between two hierarchies.')
    else:
        r1=root1.truncated(truncation_height)
        r2=root2.truncated(truncation_height)
        return r1.entropy()-r2.entropy()

## Discrepancy in #-clusters over all levels
def dNC(root1,root2,truncation_height=0):
    if root1._underlying!=root2._underlying:
        raise Exception('Underlying sets must match to compute the distance between two hierarchies.')
    else:
        return sum(abs(len(root1.slice(height))-len(root2.slice(height))) for height in root1.heights() if height>=truncation_height)


# name dictionary for hierarchical dissimilarities used in poisoning
DISTS={
    'ClusterAdjacency':dCA,
    'WorstCut':dWC,
    'MostFusion': dNC,
    #'dRobinsonFoulds':dRF,
    #'dClusterCardinality':dCC,
    #'dCrossing':dCM
}

# name dictionary for discrepancy measures used to study consecutive 
# instances of the data base
DIFFS={
    'DHdiff': dDH,
    'NCdiff': dNC,
    'ClusterAdjacency':dCA,
    'WorstCut':dWC,
}

METRICS={
    'ellone_Boolean': (
            lambda points_list: pdist(points_list,metric='cityblock'),
            lambda pt1,pt2: ellone(pt1,pt2)
        ),
    'elltwo': (
            lambda points_list: pdist(points_list,metric='euclidean'),
            lambda pt1,pt2: elltwo(pt1,pt2)
        ),
    'elltwo_normalized': (
            lambda points_list: pdist(points_list,metric='euclidean'),
            lambda pt1,pt2: elltwo(pt1,pt2)
        )
    }


### DEFINE THE CLUSTERING METHOD WHICH PRODUCES THE DIAGRAM:
def my_linkage(list_of_points,met):
    met_matr,met_func=METRICS[met]
    return linkage(list_of_points,method='single',metric=met_func)

### CONSTRUCT A MIDPOINT BETWEEN TWO POINTS
def midpoint(pt1,pt2,met):
    if met not in METRICS:
        raise Exception('Unspecified metric -- ABORTING!!!')
    else:
        #compute bridges on a case-by-case basis
        if met=='ellone_Boolean': # assuming the input points are Boolean (0/1) vectors
            mask=boolsum(pt1,pt2)
            choose_half(mask) 
            mid_pt=boolsum(pt1,mask)
        if met=='elltwo':
            mid_pt=0.5*(pt1+pt2)
        if met=='elltwo_normalized':
            mid_pt=0.5*(pt1+pt2) #(1./(2*elltwo(0.5*(pt1+pt2),0)))*(pt1+pt2)
        return mid_pt


### LIST OF "BRIDGES" IN AN MST OF $list_of_points$
def bridges(list_of_points,met):
    #pick and compute the relevant metric
    if met not in METRICS:
        raise Exception('Unspecified metric -- ABORTING!!!')
    else:
        met_matr,met_func=METRICS[met]

    #form a minimum spanning tree        
    mst=minimum_spanning_tree(csr_matrix(squareform(met_matr(list_of_points)))).todok().keys()

    #form and return the list of "bridges" -- midpoints of MST edges in the appropriate context 
    #                                         defined by the input, $met$.
    bridge_list=[]
    for i,j in mst:
        bridge_list.append((
            midpoint(list_of_points[i],list_of_points[j],met),
            met_func(list_of_points[i],list_of_points[j])
            ))
    return bridge_list



### RUN TESTS
def main():
    points=np.array([[0,0],[2,0],[3,0],[0,2],[0,-3],[7,0],[7,3],[7,1],[7,-2],[9,-1]],dtype=float)
    print 'Input point cloud:\n'
    print points
    print '\nDiameter of point cloud:\t'+str(max(dmatrix))+'\n'
    link=linkage(points) #linkage can also take the output of pdist(points) -- a flattened distance matrix -- as it input
    root=linkage_to_dendrogram(link)
    maxbits=lg(root.card())
    print 'Minimal spanning tree edges from given list of points:\n'
    print minimum_spanning_tree(csr_matrix(squareform(dmatrix))).todok().keys()
    print '\n'
    print 'Minimal spanning tree midpoints from given list of points:\n'
    print bridges(points,'elltwo')
    print '\n\n'
    print 'Producing dendrogram from linkage matrix:\n'
    print root
    print root.heights()
    
    print 'The entropy of this dendrogram is: '+str(root.entropy())+', which is about'+str(np.floor(100*root.entropy()/maxbits))+'\% of max entropy\n\n'
    print '\n\nNow combing the dendrogram:\n'
    root.comb()
    print root
    print root.heights()
    print 'The entropy of the combed dendrogram is: '+str(root.entropy())+', which is about'+str(np.floor(100*root.entropy()/maxbits))+'\% of max entropy\n\n'
    print 'Here are some slices of the dendrogram:\n'
    print 'At height 5: '
    print root.slice(4)
    print double(root.slice_incidence_matrix(4))
    print double(root.clusters_incidence_matrix())
    print 'At height 3.5: '
    print root.slice(2.5)
    print double(root.slice_incidence_matrix(2.5))
    print double(root.clusters_incidence_matrix())
    print 'At height 1.5: '
    print root.slice(.5)
    print double(root.slice_incidence_matrix(.5))
    print double(root.clusters_incidence_matrix())

    res=[1,3,4,5,6,7,9]
    print '\nRestriction to '+str(res)+' yields the dendrogram:\n'
    new_root=root.restricted(res)
    print new_root
    print new_root.heights()

    delta=1.8
    print '\nTruncation at delta='+str(delta)+' yields the dendrogram:\n'
    trunc_root=root.truncated(delta)
    print trunc_root
    print trunc_root.heights()
    print trunc_root.slice(0)
    print double(trunc_root.slice_incidence_matrix(0))
    print double(trunc_root.clusters_incidence_matrix())


    x=np.array([1,0,2,3,0,0,4,0,0,5,6])
    print x
    choose_half(x)
    print x

if __name__=="__main__":
    main()
