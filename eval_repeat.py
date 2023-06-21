import numpy as np
from tqdm import tqdm
from functools import partial
from cgdataset import World
#from independent_set import solve_lovasz_sdp, solve_max_independent_set_integer, solve_max_independent_set_binary_quad_GW, DoubleGreedy,DoubleGreedyPartialVisbilityGraph
from pydrake.geometry.optimization import (
    HPolyhedron, VPolytope, Iris, IrisOptions, Hyperellipsoid)
from region_generation import generate_regions_multi_threading
#from visibility_graphs import get_visibility_graph, create_visibility_graph_w_region_obstacles
from utils import load_experiment
import matplotlib.pyplot as plt
from visibility_seeding import VisSeeder
from seeding_utils import point_in_regions, vis_reg, shrink_regions, sorted_vertices
import shapely
from shapely.ops import cascaded_union
from scipy.sparse import lil_matrix
from vislogging import Logger
import os


seed = 1 
np.random.seed(seed)
#extract_small_examples(2000)
small_polys = []
with open("./data/small_polys.txt") as f:
	for line in f:
		small_polys.append(line.strip())
favorite_polys = small_polys
names = os.listdir('data/evalexamples')
names.sort()
for instance in names[0:6]:
	seed = 1 
	np.random.seed(seed)
	world_name = instance #"cheese102.instance.json"#small_polys[1] #"cheese205.instance.json"#fpg-poly_0000000060_h1.instance.json"#"srpg_iso_aligned_mc0000172.instance.json"##"fpg-poly_0000000070_h1.instance.json"
	world = World("./data/evalexamples/"+world_name)

	def sample_cfree_handle(n, m, regions=None):
		points = np.zeros((n,2))
		if regions is None: regions = []		
		for i in range(n):
			bt_tries = 0
			while bt_tries<m:
				point = world.sample_cfree(1)[0]
				if point_in_regions(point, regions):
					bt_tries += 1
					if bt_tries == m:
						return points, True 
				else:
					break
			points[i] = point
		return points, False

	def vgraph_builder(points, regions):
		n = len(points)
		adj_mat = lil_matrix((n,n))
		for i in tqdm(range(n)):
			point = points[i, :]
			for j in range(len(points[:i])):
				other = points[j]
				if vis_reg(point, other, world, regions):
					adj_mat[i,j] = adj_mat[j,i] = 1
		return adj_mat.toarray()

	def iris_w_obstacles(points, region_obstacles):
		regions, _ = generate_regions_multi_threading(points, world.obstacle_triangles + region_obstacles, world.iris_domain)
		return regions

	alpha = 0.05
	eps = 0.02
	N = 1
	logger = Logger(world, world_name+"_single", seed, N, alpha, eps)
	VS = VisSeeder(N = N,
				alpha = alpha,
				eps = eps,
				max_iterations = 300,
				sample_cfree = sample_cfree_handle,
				build_vgraph = vgraph_builder,
				iris_w_obstacles = iris_w_obstacles,
				verbose = True,
				logger = logger
				)

	regions = VS.run()

print('done')





