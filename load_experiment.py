import numpy as np
from tqdm import tqdm
from functools import partial
from cgdataset import World, sorted_vertices
from independent_set import solve_lovasz_sdp, solve_max_independent_set_integer, solve_max_independent_set_binary_quad_GW, DoubleGreedy,DoubleGreedyPartialVisbilityGraph
from pydrake.geometry.optimization import (
    HPolyhedron, VPolytope, Iris, IrisOptions, Hyperellipsoid)
from region_generation import generate_regions_multi_threading, fill_remaining_space, generate_region_with_region_obstacles
from visibility_graphs import get_visibility_graph
import matplotlib.pyplot as plt
from utils import experiment_string, create_experiment_directory, dump_experiment_results, load_experiment
import time
import os
import shapely
from shapely.ops import cascaded_union
# 1: Visibility graph + Integer Program, 
# 2: SDP relaxation + rounding, 
# 3: Double Greedy 
# 4: Double Greedy with partial visibility graph construction

#APPROACH = 2
PLOT_EDGES = 1

# seed = 0 
# n = 450
small_polys = []
with open("./data/small_polys.txt") as f:
	for line in f:
		small_polys.append(line.strip())
world_name = small_polys[0]

for world_name in small_polys:
	world_experiments_strings = os.listdir("./experiment_logs/"+world_name[:-5])
	world_experiments = []
	num_regions = []
	computation_time = []
	coverage = []
	
	for exp_string in world_experiments_strings:
		parts = exp_string[:-4].split('_')
		approach = parts[-1]
		seed = parts[-2]
		n = parts[-3]
		world_experiments.append([n, seed, approach])
		experiment_name = experiment_string(world_name, n, seed, approach)
		world = World("./data/examples_01/"+world_name)
		points, adj_mat, edge_endpoints, chosen_verts, regions, timings = load_experiment(world_name[:-5]+'/'+experiment_name+'.log',
																			world_name,
																			world,
																			n,
																			seed)

		shapely_regions = []
		for r in regions:
			verts = sorted_vertices(VPolytope(r))
			shapely_regions.append(shapely.Polygon(verts.T))
		comptime = timings[1]
		union_of_Polyhedra = cascaded_union(shapely_regions)
		coverage_experiment = union_of_Polyhedra.area/world.cfree_polygon.area
		
		computation_time.append(comptime)
		num_regions.append(len(regions))
		coverage.append(coverage_experiment)


	fig, axs = plt.subplots(nrows = 1, ncols=3)
	data = [computation_time, num_regions, coverage]
	names = ['time [s]', 'num regions', 'coverage cfree']
	n = 450
	seed = 0
	for ax, data_col, name_col in zip(axs, data, names):
		nfields = 3
		xpos = np.arange(nfields)
		labels = ['integer','sdp + rounding', 'double greedy']
		data_mean = [0]*3
		data_std = [0]*3
		for exp, d_exp in zip(world_experiments, data_col):
			if int(exp[0]) == n and int(exp[1]) == seed:
				data_mean[int(exp[-1])-1] = d_exp

		ax.bar(xpos, data_mean, yerr= data_std , color=['#1f77b4','#ff7f0e', '#2ca02c'], ecolor='black', capsize=20)
		ax.set_ylabel(name_col)
		ax.set_xticks(xpos)
		ax.set_xticklabels(labels)
		#shapely.plotting.plot_polygon(union_of_Polyhedra, ax=ax, add_points=True)
	plt.show()
	plt.show()


# experiment_name = experiment_string(world_name, n, seed, APPROACH)
# world = World("./data/examples_01/"+world_name)
# points, adj_mat, edge_endpoints, chosen_verts, regions, timings = load_experiment(world_name[:-5]+'/'+experiment_name+'.log',
# 								    								world_name,
# 																	world,
# 																	n,
# 																	seed)

# adj_mat = adj_mat.toarray()

# fig, ax = plt.subplots()
# world.plot_cfree(ax)
# plt.draw()
# plt.pause(0.001)
# ax.scatter(points[:, 0 ], points[:,1], color="black", s = 2)
# ax.scatter(chosen_verts[:, 0 ], chosen_verts[:,1], color="red", s = 10)
# if PLOT_EDGES:
# 	for e in edge_endpoints:
# 		ax.plot(e[0], e[1], color="black", linewidth=0.25, alpha = 0.1)
# for r in regions:
# 	world.plot_HPoly(ax, r)

# plt.draw()
# plt.pause(0.001)

# shapely_regions = []
# for r in regions:
# 	verts = sorted_vertices(VPolytope(r))
# 	shapely_regions.append(shapely.Polygon(verts.T))


# iris_with_region_obstacles_handle = partial(generate_region_with_region_obstacles, obstacles = world.obstacle_triangles, domain=world.iris_domain)

# regions, seed_points, seed_point_index = fill_remaining_space(ax, points, chosen_verts, adj_mat, regions, iris_with_region_obstacles_handle)

# shapely_regions = []
# for r in regions:
# 	verts = sorted_vertices(VPolytope(r))
# 	shapely_regions.append(shapely.Polygon(verts.T))


# union_of_Polyhedra = cascaded_union(shapely_regions)
# shapely.plotting.plot_polygon(union_of_Polyhedra, ax=ax, add_points=True)

# for r in regions:
# 	world.plot_HPoly(ax, r)
# plt.draw()
# plt.pause(0.001)
# print('')
# # 	# #
# # 	# # sample
# #     # samples_outside_regions = vs.samples_outside_regions
# #     # regions = vs.regions
# #     # connectivity_graph = vs.connectivity_graph
# #     # key_max = ''
# #     # max_vis_components = -1
# #     # #get connected components
# #     # components = [list(a) for a in nx.connected_components(connectivity_graph)]
# #     # #check if all nodes to connect are part of a single connected component
# #     # nodes_to_connect = vs.nodes_to_connect if len(vs.nodes_to_connect) else vs.guard_regions
# #     # for c in components:
# #     #     if set(nodes_to_connect) & set(c) == set(nodes_to_connect):
# #     #         return None, True



# # #extract_small_examples(2000)


# # favorite_polys = small_polys
# # #world_name = favorite_polys[0]
# # for n in [200, 300, 350, 400, 450, 500, 550]:
# # 	for world_name in favorite_polys:

# # 		experiment_name = experiment_string(world_name, n, seed, APPROACH)
# # 		prerun_experiments = os.listdir("./experiment_logs/"+world_name[:-5])
# # 		if experiment_name+".log" in prerun_experiments:
# # 			print("experiment already run")
# # 		else:
# # 			print("running experiment")
# # 			create_experiment_directory(world_name, n, seed)
# # 			world = World("./data/examples_01/"+world_name)

# # 			fig, ax = plt.subplots()
# # 			world.plot_cfree(ax)
# # 			#world.plot_triangles(ax)
# # 			plt.draw()
# # 			plt.pause(0.001)
# # 			#plt.waitforbuttonpress()
# # 			if APPROACH !=4:
# # 				points, adj_mat, edge_endpoints, twgen, tgraphgen = get_visibility_graph(world_name, world, n, seed) 
# # 				adj_mat = adj_mat.toarray()
# # 				ax.scatter(points[:, 0 ], points[:,1], color="black", s = 2)
# # 				if PLOT_EDGES:
# # 					for e in edge_endpoints:
# # 						ax.plot(e[0], e[1], color="black", linewidth=0.25, alpha = 0.1)
# # 				plt.draw()
# # 				plt.pause(0.001)

# # 			#theta, mat = solve_lovasz_sdp(adj_mat)
# # 			#print("Lovasz ")
# # 			#print(theta)
# # 			#print(mat)
# # 			t0 = time.time()
# # 			if APPROACH ==1:
# # 				m, verts = solve_max_independent_set_integer(adj_mat)
# # 				#print('Independent set sol', m)
# # 				chosen_verts = points[np.nonzero(verts)]
# # 				print("Integer Program Solution")
# # 				print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# # 			elif APPROACH ==2:
# # 				m, verts = solve_max_independent_set_binary_quad_GW(adj_mat, n_rounds=1000)
# # 				chosen_verts = points[np.nonzero(verts)]
# # 				print("Binary Quadratic SDP Relaxation + Rounding Solution")
# # 				print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# # 				#ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
# # 				#ax.scatter(points[:,0], points[:,1], color="black", s = 1)
# # 				plt.draw()

# # 			elif APPROACH ==3:
# # 				dg = DoubleGreedy(Vertices = points,
# # 								Adjacency_matrix=adj_mat,
# # 								verbose=True,
# # 								seed = seed)
# # 				chosen_verts = np.array(dg.construct_independent_set())
# # 				# violations = 0
# # 				# for i in dg.independent_set:
# # 				# 	for j in dg.independent_set:
# # 				# 		if i!=j:
# # 				# 			violations += adj_mat[i,j]
# # 				# print("violations", violations)
# # 				# print("Initial Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# # 				chosen_verts = np.array(dg.refine_independent_set_greedy())
# # 				# violations = 0
# # 				# for i in dg.independent_set:
# # 				# 	for j in dg.independent_set:
# # 				# 		if i!=j:
# # 				# 			violations += adj_mat[i,j]
# # 				# print("violations", violations)
# # 				#ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red", s = 15)
# # 				#plt.pause(0.1)
# # 				print("Double Greedy Solution")
# # 				print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
# # 			elif APPROACH ==4:
# # 				def sample_node(w):
# # 					return w.sample_cfree(1)[0]

# # 				sample_node_handle = partial(sample_node, w = world)
# # 				dg = DoubleGreedyPartialVisbilityGraph(alpha = 0.001,
# # 								eps = 0.001,
# # 								max_samples = n,
# # 								sample_node_handle=sample_node_handle,
# # 								los_handle = world.visible,
# # 								verbose=True)
# # 				chosen_verts = np.array(dg.construct_independent_set())
# # 				chosen_verts = np.array(dg.refine_independent_set_greedy())
# # 				# v_sampleset = np.array([dg.sample_set[k][0] for k in dg.sample_set.keys()])
# # 				# ax.scatter(v_sampleset[:,0], v_sampleset[:,1], color="k", s = 1)
# # 				print("Double Greedy Solution Partial Visibility Graph")
# # 				print("Hidden Set Size: ", len(chosen_verts), " in ", len(dg.points), " samples")
# # 			else:
# # 				raise NotImplementedError("Choose valid approach {1,2,3,4}, chosen", APPROACH)
# # 			t1 = time.time()
# # 			ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")	
# # 			plt.draw()
# # 			plt.pause(0.01)
# # 			#plt.waitforbuttonpress()
# # 			regions, seed_points = generate_regions_multi_threading(chosen_verts, world.obstacle_triangles, world.iris_domain)
# # 			print('done generating regions')
# # 			for r in regions:
# # 				world.plot_HPoly(ax, r)
# # 			t2 = time.time()
# # 			dump_experiment_results(world_name, experiment_name, chosen_verts, regions, t1-t0, t2-t1)
# # #plt.draw()
# # #plt.show()
# # #plt.waitforbuttonpress()
# # print('done')