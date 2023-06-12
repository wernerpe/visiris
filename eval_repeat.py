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
from seeding_utils import point_in_regions, vis_reg, shrink_regions
from scipy.sparse import lil_matrix

seed = 0 
np.random.seed(seed)
#extract_small_examples(2000)
small_polys = []
with open("./data/small_polys.txt") as f:
	for line in f:
		small_polys.append(line.strip())
favorite_polys = small_polys
world_name = "cheese102.instance.json"#small_polys[1] #"cheese205.instance.json"#fpg-poly_0000000060_h1.instance.json"#"srpg_iso_aligned_mc0000172.instance.json"##"fpg-poly_0000000070_h1.instance.json"
world = World("./data/examples_01/"+world_name)

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

fig, ax = plt.subplots(figsize = (10,10))
world.plot_cfree(ax)

# pt,_ =  sample_cfree_handle(2,1000, [])
# pt = np.array([[2.87, 8.02],[7.41, 4.9]])
# ax.scatter(pt[:,0], pt[:,1], c = 'r')
# regions = iris_w_obstacles(pt, shrink_regions([],0.25))#generate_regions_multi_threading(pt, world.obstacle_triangles + shrink_regions([], 0.25), world.iris_domain)
# for r in regions:
# 	world.plot_HPoly(ax, r, color = 'r')
# plt.pause(0.01)

VS = VisSeeder(N = 200,
	       	   alpha = 0.05,
			   eps = 0.05,
			   max_iterations = 10,
			   sample_cfree = sample_cfree_handle,
			   build_vgraph = vgraph_builder,
			   iris_w_obstacles = iris_w_obstacles,
			   verbose = True,
	       		)
regions = VS.run()
for g in VS.region_groups:
	rnd_artist = ax.plot([0,0],[0,0], alpha = 0)
	for r in g:
		world.plot_HPoly(ax, r, color =rnd_artist[0].get_color())
plt.show()


print('done')







# n = 250
# APPROACH = 3
# PLOT_EDGES = 1



# # #world.plot_triangles(ax)
# fig, ax = plt.subplots(figsize = (10,10))
# world.plot_cfree(ax)
# points, adj_mat, edge_endpoints = create_visibility_graph_w_region_obstacles(world, n, seed, None)
# adj_mat = adj_mat.toarray()
# ax.scatter(points[:, 0 ], points[:,1], color="black", s = 2)
# if PLOT_EDGES:
# 	for e in edge_endpoints[::10]:
# 		ax.plot(e[0], e[1], color="black", linewidth=0.25, alpha = 0.1)

# dg = DoubleGreedy(Vertices = points,
# 				Adjacency_matrix=adj_mat,
# 				verbose=True,
# 				seed = seed)
# chosen_verts = np.array(dg.construct_independent_set())
# chosen_verts = np.array(dg.refine_independent_set_greedy())
# print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# regions, seed_points = generate_regions_multi_threading(chosen_verts, world.obstacle_triangles, world.iris_domain)
# rnd_artist = ax.plot([0,0],[0,0], alpha = 0)
# for r in regions:
# 	world.plot_HPoly(ax, r, color =rnd_artist[0].get_color())
# plt.draw()
# plt.pause(0.001)
# region_obstacles = shrink_regions(regions)
# points2, adj_mat2, edge_endpoints = create_visibility_graph_w_region_obstacles(world, n, seed, regions, shrunken_regions = region_obstacles)
# adj_mat2 = adj_mat2.toarray()

# dg = DoubleGreedy(Vertices = points2,
# 				Adjacency_matrix=adj_mat2,
# 				verbose=True,
# 				seed = seed)
# chosen_verts2 = np.array(dg.construct_independent_set())
# chosen_verts2 = np.array(dg.refine_independent_set_greedy())
# print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# regions2, seed_points2 = generate_regions_multi_threading(chosen_verts2, world.obstacle_triangles + region_obstacles, world.iris_domain)
# rnd_artist = ax.plot([0,0],[0,0], alpha = 0)
# for r in regions2:
# 	world.plot_HPoly(ax, r, color =rnd_artist[0].get_color())

# ax.scatter(points2[:, 0 ], points2[:,1], color=rnd_artist[0].get_color(), s = 10)
# if PLOT_EDGES:
# 	for e in edge_endpoints[::10]:
# 		ax.plot(e[0], e[1],  linewidth=0.25, alpha = 0.1, color = rnd_artist[0].get_color())

# plt.draw()
# plt.pause(0.001)
# print('here')
# #theta, mat = solve_lovasz_sdp(adj_mat)
# #print("Lovasz ")
# #print(theta)
# #print(mat)
# # if APPROACH ==1:
# # 	m, verts = solve_max_independent_set_integer(adj_mat)
# # 	print("Integer Program Solution")
# # 	#print('Independent set sol', m)
# # 	chosen_verts = points[np.nonzero(verts)]
# # 	print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# # 	ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
# # 	ax.scatter(points[:,0], points[:,1], color="black", s = 1)
	
# # 	plt.draw()
# # 	plt.waitforbuttonpress()

# # elif APPROACH ==2:
# # 	m, verts = solve_max_independent_set_binary_quad_GW(adj_mat, n_rounds=1000)
# # 	print("Binary Quadratic SDP Relaxation + Rounding Solution")
# # 	chosen_verts = points[np.nonzero(verts)]
# # 	print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# # 	ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
# # 	ax.scatter(points[:,0], points[:,1], color="black", s = 1)
# # 	plt.draw()

# if APPROACH ==3:
# 	dg = DoubleGreedy(Vertices = points,
# 		   			  Adjacency_matrix=adj_mat,
# 					  verbose=True,
# 					  seed = seed)
# 	chosen_verts = np.array(dg.construct_independent_set())
# 	chosen_verts = np.array(dg.refine_independent_set_greedy())
# 	print("Initial Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# 	regions, seed_points = generate_regions_multi_threading(chosen_verts, world.obstacle_triangles, world.iris_domain)
# 	rnd_artist = ax.scatter([0],[0], opacity = 0)
# 	for r in regions:
# 		world.plot_HPoly(ax, r, color = rnd_artist[0].get_color())
	



# 	# violations = 0
# 	# for i in dg.independent_set:
# 	# 	for j in dg.independent_set:
# 	# 		if i!=j:
# 	# 			violations += adj_mat[i,j]
# 	# print("violations", violations)
# 	# ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red", s = 15)
# 	plt.pause(0.1)
# 	print("Double Greedy Solution")
# 	print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# # elif APPROACH ==4:
# # 	def sample_node(w):
# # 		return w.sample_cfree(1)[0]

# # 	sample_node_handle = partial(sample_node, w = world)
# # 	dg = DoubleGreedyPartialVisbilityGraph(alpha = 0.001,
# # 					eps = 0.001,
# # 					max_samples = n,
# # 					sample_node_handle=sample_node_handle,
# # 					los_handle = world.visible,
# # 					verbose=True)
# # 	chosen_verts = np.array(dg.construct_independent_set())
# # 	chosen_verts = np.array(dg.refine_independent_set_greedy())
# # 	ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
# # 	v_sampleset = np.array([dg.sample_set[k][0] for k in dg.sample_set.keys()])
# # 	ax.scatter(v_sampleset[:,0], v_sampleset[:,1], color="k", s = 1)
# # 	print("Double Greedy Solution")
# # 	print("Hidden Set Size: ", len(chosen_verts), " in ", len(dg.points), " samples")
# # else:
# # 	raise NotImplementedError("Choose valid approach {1,2,3,4}, chosen", APPROACH)
# print('done generating regions')
# for r in regions:
# 	world.plot_HPoly(ax, r)

# plt.draw()


# plt.show()
# plt.waitforbuttonpress()
# print('done')