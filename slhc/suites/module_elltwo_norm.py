### module for basic operations with elltwo_normalized metric on Boolean feature vectors
from scipy.spatial.distance import euclidean as elltwo
from scipy.spatial.distance import pdist

### normalize a boolean feature vector to the Euclidean unit sphere
def normalize2(x):
    try:
        return (1./elltwo(x,0))*x
    except:
        x

### We treat the vectors as normalized to the unit sphere
def get_metric(params=None):
    met_func=lambda pt1,pt2: elltwo(normalize2(pt1),normalize2(pt2)) if params is None else lambda pt1,pt2: elltwo(normalize2(params*pt1),normalize2(params*pt2))
    matr_func=lambda points_list: pdist([normalize2(pt) for pt in points_list],metric='euclidean') if params is None else lambda points_list: pdist([normalize2(params*pt) for pt in points_list],metric='euclidean')
    return met_func,matr_func

### Construct a midpoint ON THE UNIT SPHERE
def midpoint(pt1,pt2):
    return normalize2(normalize2(pt1)+normalize2(pt2))

### MUST CONSIDER A MORE SYMMETRIC CONSTRUCTION, mapping x to normalize2(2*x-1),
### but then there are questions regarding the generation of midpoints.