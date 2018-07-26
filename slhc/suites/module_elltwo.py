### module for basic operations with elltwo metric
from scipy.spatial.distance import euclidean as elltwo
from scipy.spatial.distance import pdist

def get_metric(params=None):
    met_func=lambda pt1,pt2: elltwo(pt1,pt2) if params is None else lambda pt1,pt2: elltwo(params*pt1,params*pt2)
    matr_func=lambda points_list: pdist(points_list,metric='euclidean') if params is None else lambda points_list: pdist([params*pt for pt in points_list],metric='euclidean')
    return met_func,matr_func

def midpoint(pt1,pt2):
    return 0.5*(pt1+pt2)

