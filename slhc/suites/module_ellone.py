### module for basic operations with ellone metric
from scipy.spatial.distance import cityblock as ellone
from scipy.spatial.distance import pdist

def get_metric(params=None):
    met_func=lambda pt1,pt2: ellone(pt1,pt2) if params is None else lambda pt1,pt2: ellone(params*pt1,params*pt2)
    matr_func=lambda points_list: pdist(points_list,metric='cityblock') if params is None else lambda points_list: pdist([params*pt for pt in points_list],metric='cityblock')
    return met_func,matr_func

def midpoint(pt1,pt2):
    return 0.5*(pt1+pt2)

