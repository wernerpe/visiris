from cgdataset import World
import matplotlib.pyplot as plt
import numpy as np
from seeding_utils import point_in_regions, point_near_regions, vis_reg, compute_kernels, sorted_vertices
from independent_set_solver import solve_max_independent_set_integer
from scipy.sparse import lil_matrix
from tqdm import tqdm
from shapely.ops import cascaded_union
import shapely
from region_generation import generate_regions_ellipses_multi_threading
from doublegreedyhiddenset import DoubleGreedySeeding
from pydrake.geometry.optimization import (
    HPolyhedron, VPolytope, Iris, IrisOptions, Hyperellipsoid)
from functools import partial
from vislogging import Logger
from time import strftime, gmtime

eps_sample = -0.05
eps = 0.05
alpha = 0.1
N=1
seed = 3
use_region_obstacles_iris = 1
#use_region_visibility_obstacles = 0
region_pullback=0.25

world_name = "cheese102.instance.json"#small_polys[1] #"cheese205.instance.json"#fpg-poly_0000000060_h1.instance.json"#"srpg_iso_aligned_mc0000172.instance.json"##"fpg-poly_0000000070_h1.instance.json"
world = World("./data/evalexamples/"+world_name)
world.build_offset_cfree(eps_sample)
np.random.seed(seed)
def sample_cfree_handle(n, m, regions=None):
	points = np.zeros((n,2))
	if regions is None: regions = []		
	for i in range(n):
		bt_tries = 0
		while bt_tries<m:
			point = world.sample_cfree_distance(1, eps = eps_sample)[0]
			#point = world.sample_cfree(1)[0]
			if point_near_regions(point, regions, tries = 100, eps = 0.1):
				bt_tries+=1
			else:
				break
		if bt_tries == m:
			return points, True
		
		points[i] = point
	return points, False


def mut_visreg(point, other, regions, world):
	return vis_reg(point.squeeze(), other.squeeze(), world, regions, n_checks=100)

los_handle = partial(mut_visreg, world = world)

def compute_coverage(regions):
    shapely_regions = []
    for r in regions:
        verts = sorted_vertices(VPolytope(r))
        shapely_regions.append(shapely.Polygon(verts.T))
    union_of_Polyhedra = cascaded_union(shapely_regions)
    return union_of_Polyhedra.area/world.cfree_polygon.area

def compute_coverage_cfree_tot(regions):
    shapely_regions = []
    for r in regions:
        verts = sorted_vertices(VPolytope(r))
        shapely_regions.append(shapely.Polygon(verts.T))
    union_of_Polyhedra = cascaded_union(shapely_regions)
    return union_of_Polyhedra.area/world.cfree_polygon.area

def iris_w_obstacles(points, ellipsoids, region_obstacles, old_regions = None, use_region_obstacles = use_region_obstacles_iris):
    if N>=1:
        #+ region_obstacles
        obstacles = [r for r in world.obstacle_triangles]
        if use_region_obstacles:
            obstacles += region_obstacles
        regions, _, is_full = generate_regions_ellipses_multi_threading(points, ellipsoids, obstacles, world.iris_domain, compute_coverage, coverage_threshold=1-eps, old_regs = old_regions)
    else:
        #if N=1 coverage estimate happens at every step
        obstacles = [r for r in world.obstacle_triangles]
        if use_region_obstacles:
            obstacles += region_obstacles
        regions, _, _ = generate_regions_ellipses_multi_threading(points, ellipsoids, obstacles, world.iris_domain)
        is_full = 1-eps <= compute_coverage(old_regions+regions)
    return regions, is_full


logger = Logger(world, world_name, f"_DG_noobs_{use_region_obstacles_iris}_{eps_sample}_{region_pullback}_"+strftime("%m_%d_%H_%M_%S_", gmtime()), seed, N, alpha, eps)

VS = DoubleGreedySeeding(N = N,
						alpha = alpha,
						eps = eps,
						max_iterations = 300,
						sample_cfree = sample_cfree_handle,
						los_handle= los_handle,
						iris_w_obstacles = iris_w_obstacles,
						verbose = True,
						logger = logger,
						region_pullback = region_pullback,
						use_kernelseeding=True,
                        seed = seed
						)

regions = VS.run()