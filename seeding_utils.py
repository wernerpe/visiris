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

def vis_reg(point, other, world, regions):
    if not world.visible(point, other):
        return False
    else:
        tval = np.linspace(0, 1, 40)
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