import numpy as np
from tqdm import tqdm
from functools import partial
from cgdataset import World, sorted_vertices
from old_scripts.independent_set import solve_lovasz_sdp, solve_max_independent_set_integer, solve_max_independent_set_binary_quad_GW, DoubleGreedy,DoubleGreedyPartialVisbilityGraph
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
import re

def extract_string_before_first_digit(string):
	match = re.search(r"\d", string)
	if match:
		name_head = string[:match.start()]
		is_zeros = string[match.start():match.start()+3] == '000'
		number = string[match.start()+match.end()-1-3:match.start()+match.end()-1] if is_zeros else string[match.start():match.start()+3]
		name_head = name_head if len(name_head)<6 else name_head[:7]
		name = name_head + '_' + number
		return name  
	else:
		return string
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

experiments = {}
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
		comptime = timings[1] + timings[0]
		union_of_Polyhedra = cascaded_union(shapely_regions)
		coverage_experiment = union_of_Polyhedra.area/world.cfree_polygon.area
		
		computation_time.append(comptime)
		num_regions.append(len(regions))
		coverage.append(coverage_experiment)

	experiments[exp_string] = [computation_time, num_regions, coverage, world_experiments]

	#fig, axs = plt.subplots(nrows = 1, ncols=3, figsize = (10, 8))
	#data = [computation_time, num_regions, coverage]
	# for ax, data_col, name_col in zip(axs, data, names):
	# 	nfields = 3
	# 	xpos = np.arange(nfields)
	# 	labels = ['integer','sdp + rounding', 'double greedy']
	# 	data_mean = [0]*3
	# 	data_std = [0]*3
	# 	for exp, d_exp in zip(world_experiments, data_col):
	# 		if int(exp[0]) == n and int(exp[1]) == seed:
	# 			data_mean[int(exp[-1])-1] = d_exp

	# 	ax.bar(xpos, data_mean, yerr= data_std , color=['#1f77b4','#ff7f0e', '#2ca02c'], ecolor='black', capsize=20)
	# 	ax.set_ylabel(name_col)
	# 	ax.set_xticks(xpos)
	# 	ax.set_xticklabels(labels)
	# 	#shapely.plotting.plot_polygon(union_of_Polyhedra, ax=ax, add_points=True)
n = 450
seed = 0

fig, axs = plt.subplots(nrows = 1, ncols=3, figsize = (15, 8))
names = ['time [s]', 'number hidden points / regions', 'coverage cfree']
for idx, ax in enumerate(axs):
	ax.set_xticks(np.arange(len(small_polys)))
	ax.set_xticklabels([extract_string_before_first_digit(s[:-14]) for s in small_polys], rotation = 45,  ha='right')
	ax.set_ylabel(names[idx], fontsize = 14)
	data_integer = [0]*len(small_polys)
	data_sdp = [0]*len(small_polys)
	data_greedy = [0]*len(small_polys)
	data_greedy_partial = [0]*len(small_polys)

	for idx_world, (key, exp) in enumerate(experiments.items()):
		data = exp[idx]
		infos = exp[-1]
		for dat,  info in zip(data, infos):
			nstr, seedstr, approach_str = info
			if int(nstr) == n:
				if int(approach_str) == 1:
					data_integer[idx_world] = dat
				elif int(approach_str) ==2:
					data_sdp[idx_world] = dat
				elif int(approach_str) ==3:
					data_greedy[idx_world] = dat
				elif int(approach_str) ==4:
					data_greedy_partial[idx_world] = dat
				
	ax.plot(data_integer, linewidth=0.5, marker = 'o', label = 'integer', c = 'k')
	ax.plot(data_sdp, label = 'sdp', linewidth=0.5, marker = '^', c = 'r')
	ax.plot(data_greedy, label = 'double greedy',linewidth=0.5, marker = 'x', c = 'g')
	ax.plot(data_greedy_partial, label = 'double greedy partial',linewidth=0.5, marker = 's', c = 'b')
	ax.legend()
	fig.suptitle(f"Visibility Graph with {n} Nodes", fontsize=16)
plt.show()

print('')
