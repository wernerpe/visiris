import numpy as np
from pydrake.all import HPolyhedron

def shrink_regions(regions, offset_fraction = 0.25):
	shrunken_regions = []
	for r in regions:
		offset = offset_fraction*np.min(1/np.linalg.eig(r.MaximumVolumeInscribedEllipsoid().A())[0])
		rnew = HPolyhedron(r.A(), r.b()-offset)
		shrunken_regions.append(rnew)	
	return shrunken_regions

def point_in_regions(pt, regions):
    for r in regions:
        if r.PointInSet(pt.reshape(-1,1)):
            return True
    return False

def point_near_regions(pt, regions, tries = 10, eps = 0.1):
    
    for _ in range(tries):
        n = 2*eps*(np.random.rand(len(pt))-0.5)
        checkpt = pt+n
        for r in regions:
            if r.PointInSet(checkpt.reshape(-1,1)):
                return True
    return False

def vis_reg(point, other, world, regions, n_checks = 50):
    if not world.visible(point, other):
        return False
    else:
        tval = np.linspace(0, 1, n_checks)
        for t in tval:
            pt = (1-t)*point + t* other
            if point_in_regions(pt, regions):
                return False
    return True

def sorted_vertices(vpoly):
    assert vpoly.ambient_dimension() == 2
    poly_center = np.sum(vpoly.vertices(), axis=1) / vpoly.vertices().shape[1]
    vertex_vectors = vpoly.vertices() - np.expand_dims(poly_center, 1)
    sorted_index = np.arctan2(vertex_vectors[1], vertex_vectors[0]).argsort()
    return vpoly.vertices()[:, sorted_index]


def assert_numpy_array_with_integer_dtype(a):
    assert isinstance(a, np.ndarray), "Input 'a' must be a NumPy array."
    assert np.issubdtype(a.dtype, np.integer), "Data type of 'a' must be integer."

def remove_element(arr, index_to_remove):
    return np.delete(arr, index_to_remove)

def compute_kernels(ind_set, adj_mat):
    assert_numpy_array_with_integer_dtype(ind_set)
    kernels = []
    for g in ind_set:
        #check for all points that can see hidden point g
        adj_points_idx = np.where(adj_mat[g,:]==1)[0]
        #assert that can only see one of the hidden points
        adj_ind_set = adj_mat[ind_set, :]
        adj_ind_set = adj_ind_set[:, adj_points_idx]

        #pick the ones that can only see g
        num_vis_hidden_points = np.sum(adj_ind_set, axis = 0)
        kernel = list(np.where(num_vis_hidden_points ==1)[0]) 
        kernels.append(np.array([adj_points_idx[k] for k in kernel]+[g]))
        
    return kernels