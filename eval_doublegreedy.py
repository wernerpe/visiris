import numpy as np
from tqdm import tqdm
from functools import partial
from cgdataset import World
#from independent_set import solve_lovasz_sdp, solve_max_independent_set_integer, solve_max_independent_set_binary_quad_GW, DoubleGreedy,DoubleGreedyPartialVisbilityGraph
from pydrake.geometry.optimization import (
    HPolyhedron, VPolytope, Iris, IrisOptions, Hyperellipsoid)
from region_generation import generate_regions_multi_threading, generate_regions_multi_threading_regobs
#from visibility_graphs import get_visibility_graph, create_visibility_graph_w_region_obstacles
from utils import load_experiment, generate_random_colors
import matplotlib.pyplot as plt
# from visibility_seeding import VisSeeder
from seeding_utils import point_near_regions, point_in_regions, vis_reg, shrink_regions, sorted_vertices
import shapely
from shapely.ops import cascaded_union
from scipy.sparse import lil_matrix
from vislogging import Logger
import os
from doublegreedyhiddenset import DoubleGreedySeeding
from time import strftime, gmtime

seed = 102
np.random.seed(seed)
#extract_small_examples(2000)
small_polys = []
with open("./data/small_polys.txt") as f:
	for line in f:
		small_polys.append(line.strip())
favorite_polys = small_polys
names = os.listdir('data/evalexamples')
names.sort()

eps_sample = -0.05
seed = 102
use_region_obstacles_iris = 1
use_region_visibility_obstacles = 0
region_pullback=0.0
instance = 'cheese102.instance.json'
N = 30000

# for instance in ['srpg_iso_aligned_mc0000172.instance.json', 'cheese102.instance.json']:#'srpg_iso_aligned_mc0000172.instance.json']:#, #,'fpg-poly_0000000060_h1.instance.json']: #'cheese102.instance.json', 
# 	for seed in [2,3,4]:
# 		for N in [1, 300]: 
np.random.seed(seed)
world_name = instance #"cheese102.instance.json"#small_polys[1] #"cheese205.instance.json"#fpg-poly_0000000060_h1.instance.json"#"srpg_iso_aligned_mc0000172.instance.json"##"fpg-poly_0000000070_h1.instance.json"
world = World("./data/evalexamples/"+world_name)
world.build_offset_cfree(eps_sample)


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

def vgraph_builder(points, regions, region_vis_obstacles=use_region_visibility_obstacles):
	n = len(points)
	adj_mat = lil_matrix((n,n))
	for i in tqdm(range(n)):
		point = points[i, :]
		for j in range(len(points[:i])):
			other = points[j]
			if region_vis_obstacles:
				if vis_reg(point, other, world, regions):
					adj_mat[i,j] = adj_mat[j,i] = 1
			else:
				if vis_reg(point, other, world, []):
					adj_mat[i,j] = adj_mat[j,i] = 1
	return adj_mat.toarray()



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

# def iris_w_obstacles(points, region_obstacles, old_regions = None, use_region_obstacles = use_region_obstacles_iris):
# 	if N>1:
# 		#+ region_obstacles
# 		obstacles = [r for r in world.obstacle_triangles]
# 		if use_region_obstacles:
# 			obstacles += region_obstacles
# 		regions, _, is_full = generate_regions_multi_threading(points, obstacles, world.iris_domain, compute_coverage, coverage_threshold=1-eps, old_regs = old_regions)
# 	else:
# 		#if N=1 coverage estimate happens at every step
# 		obstacles = [r for r in world.obstacle_triangles]
# 		if use_region_obstacles:
# 			obstacles += region_obstacles
# 		regions, _, _ = generate_regions_multi_threading(points, obstacles, world.iris_domain)
# 		is_full = 1-eps <= compute_coverage(old_regions+regions)
# 	return regions, is_full

def iris_w_obstacles(points, region_obstacles, old_regions = None, use_region_obstacles = use_region_obstacles_iris):
	if N>1:
		#+ region_obstacles
		#obstacles = [r for r in world.obstacle_triangles]
		# if use_region_obstacles:
		# 	obstacles += region_obstacles
		regions, _, is_full = generate_regions_multi_threading_regobs(points, [r for r in world.obstacle_triangles], region_obstacles, world.iris_domain, compute_coverage, coverage_threshold=1-eps, old_regs = old_regions, noregits = 3)
	else:
		#if N=1 coverage estimate happens at every step
		obstacles = [o for o in world.obstacle_triangles]
		if use_region_obstacles:
			obstacles += region_obstacles
		regions, _, _ = generate_regions_multi_threading_regobs(points, [r for r in world.obstacle_triangles], region_obstacles, world.iris_domain, noregits = 3)
		is_full = 1-eps <= compute_coverage(old_regions+regions)
	return regions, is_full

alpha = 0.05
eps = 0.1

def mut_visreg(point, other, regions, world):
	return vis_reg(point.squeeze(), other.squeeze(), world, regions, n_checks=100)

los_handle = partial(mut_visreg, world = world)

logger = Logger(world, world_name, f"_DG_{use_region_obstacles_iris}_{use_region_visibility_obstacles}_{eps_sample}_{region_pullback}_"+strftime("%m_%d_%H_%M_%S_", gmtime()), seed, N, alpha, eps)
from doublegreedyhiddenset import HiddensetDoubleGreedy

dg = HiddensetDoubleGreedy(
		alpha=alpha,
		eps = 0.0001,
		max_samples = N,
		sample_node_handle=sample_cfree_handle,
		los_handle=los_handle,
		verbose=True
		) 
            
#sregs = shrink_regions([], offset_fraction=0.25)  
dg.construct_independent_set([])

fig,ax = plt.subplots(figsize = (10,10))
# ax.set_xlim((-size, size))
# ax.set_ylim((-size, size))
#world.plot_cfree_offset(ax)
world.plot_cfree(ax)
pts = np.array(dg.points).squeeze()

ax.scatter(pts[:,0], pts[:,1], s=1, c ='k', alpha = 0.5)
# ax.scatter(pts[dg.hidden_set,0], pts[dg.hidden_set,1], s=125, c ='b', zorder = 10)

# dg.refine_independent_set_greedy([], ax)
dg.refine_independent_set_greedy([])

print("hiddenset", len(dg.hidden_set))
colors = generate_random_colors(len(dg.hidden_set))

s_shadow =70
s_color = 65
s_sample_set = 50
# ax.scatter(pts[dg.hidden_set,0], pts[dg.hidden_set,1], s=45, c ='r', zorder = 10)

kernels = [np.array(dg.compute_kernel_of_hidden_point(p) + [dg.points[p]]).reshape(-1,2) for p in dg.hidden_set]
for c, k in zip(colors, kernels):
	ax.scatter(k[:,0], k[:,1], s = s_shadow,c = 'k', zorder = 11)
	ax.scatter(k[:,0], k[:,1], s = s_color, c = [c]*len(k), zorder = 12)
	for p in k[:-1,:]:
		ax.plot([p[0], k[-1,0]], [p[1], k[-1,1]], c =c)

for hpt_idx in dg.hidden_set:
	hpt = pts[hpt_idx].reshape(1,2)
	hpt_str = str(hpt)
	vis = []
	i = 0
	ds = 5
	for key in dg.sample_set.keys():
		if i%ds ==0:
			pass
		else:
			point = dg.sample_set[key][0]
			visible_points = dg.sample_set[key][1]
			for p in visible_points:
				if str(p)==hpt_str:
					vis.append(point)
		i+=1
	vis_arr = np.array(vis).reshape(-1,2)
	ax.scatter(vis_arr[:,0], vis_arr[:,1], s = s_sample_set, color = [colors[dg.hidden_set.index(hpt_idx)]]*len(vis_arr), alpha = 0.2)

ax.scatter(pts[dg.hidden_set,0], pts[dg.hidden_set,1], s=10, c ='r', zorder = 15)
plt.pause(0.01)



print("hiddenset", len(dg.hidden_set))
plt.show(block=False)
exit()


VS = DoubleGreedySeeding(N = N,
						alpha = alpha,
						eps = eps,
						max_iterations = 300,
						sample_cfree = sample_cfree_handle,
						build_vgraph = vgraph_builder,
						iris_w_obstacles = iris_w_obstacles,
						verbose = True,
						logger = logger,
						region_pullback = region_pullback
						)

regions = VS.run()

# fig,ax = plt.subplots(figsize = (10,10))
# # ax.set_xlim((-size, size))
# # ax.set_ylim((-size, size))
# world.plot_cfree_offset(ax)

# for r in regions:
# 	world.plot_HPoly(ax, r)

# pts, _ = sample_cfree_handle(100, 5000, regions)
# ax.scatter(pts[:, 0],pts[:, 1])
# plt.show(block = False)

# sp = np.array([2.38117351, 8.13097369])
#from region_generation import generate_regions_regobs

# generate_regions_regobs([sp],
# 						[o.A() for o in world.obstacle_triangles],
# 						[o.b() for o in world.obstacle_triangles],
# 						[],
# 						[],
# 						world.iris_domain.A(),
# 						world.iris_domain.b())
#len(world.obstacle_triangles)
# print('done')





