import numpy as np
from tqdm import tqdm
from functools import partial
from cgdataset import World
from independent_set import solve_lovasz_sdp, solve_max_independent_set_integer, solve_max_independent_set_binary_quad_GW, DoubleGreedy,DoubleGreedyPartialVisbilityGraph
from pydrake.geometry.optimization import (
    HPolyhedron, VPolytope, Iris, IrisOptions, Hyperellipsoid)
from region_generation import generate_regions_multi_threading
import matplotlib.pyplot as plt

np.random.seed(0)
favorite_polys = ["srpg_iso_aligned_mc0000172.instance.json", "cheese102.instance.json", "srpg_mc0000579.instance.json", "fpg-poly_0000000070_h1.instance.json"]
world = World("./data/examples_01/"+favorite_polys[0])
n = 500
# 1: Visibility graph + Integer Program, 
# 2: SDP relaxation + rounding, 
# 3: Double Greedy 
# 4: Double Greedy with partial visibility graph construction

APPROACH = 3
PLOT_EDGES = 0

fig, ax = plt.subplots()
world.plot_cfree(ax)
#world.plot_triangles(ax)
plt.draw()
plt.pause(0.001)
#plt.waitforbuttonpress()
edge_endpoints = []
if APPROACH !=4:
	points = np.zeros((n,2))
	adj_mat = np.zeros((n,n))
	for i in tqdm(range(n)):
		point = world.sample_cfree(1)[0]
		#ax.scatter([point[0]], [point[1]], color="black", s = 2)
		for j in range(len(points[:i])):
			other = points[j]
			if world.visible(point, other):
				#ax.plot([point[0], other[0]], [point[1], other[1]], color="black", linewidth=0.25, alpha = 0.5)
				edge_endpoints.append([[point[0], other[0]], [point[1], other[1]]])
				adj_mat[i,j] = adj_mat[j,i] = 1
		points[i] = point
	ax.scatter(points[:, 0 ], points[:,1], color="black", s = 2)
	if PLOT_EDGES:
		for e in edge_endpoints:
			ax.plot(e[0], e[1], color="black", linewidth=0.25, alpha = 0.1)
	plt.draw()
	plt.pause(0.001)

#theta, mat = solve_lovasz_sdp(adj_mat)
#print("Lovasz ")
#print(theta)
#print(mat)
if APPROACH ==1:
	m, verts = solve_max_independent_set_integer(adj_mat)
	print("Integer Program Solution")
	#print('Independent set sol', m)
	chosen_verts = points[np.nonzero(verts)]
	print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
	ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
	ax.scatter(points[:,0], points[:,1], color="black", s = 1)
	
	plt.draw()
	plt.waitforbuttonpress()

elif APPROACH ==2:
	m, verts = solve_max_independent_set_binary_quad_GW(adj_mat, n_rounds=1000)
	print("Binary Quadratic SDP Relaxation + Rounding Solution")
	chosen_verts = points[np.nonzero(verts)]
	print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
	ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
	ax.scatter(points[:,0], points[:,1], color="black", s = 1)
	plt.draw()

elif APPROACH ==3:
	np.random.seed(1)
	dg = DoubleGreedy(Vertices = points,
		   			  Adjacency_matrix=adj_mat,
					  verbose=True,
					  seed = 1)
	chosen_verts = np.array(dg.construct_independent_set())
	violations = 0
	for i in dg.independent_set:
		for j in dg.independent_set:
			if i!=j:
				violations += adj_mat[i,j]
	print("violations", violations)
	print("Initial Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
	chosen_verts = np.array(dg.refine_independent_set_greedy())
	violations = 0
	for i in dg.independent_set:
		for j in dg.independent_set:
			if i!=j:
				violations += adj_mat[i,j]
	print("violations", violations)
	ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red", s = 15)
	plt.pause(0.1)
	print("Double Greedy Solution")
	print("Hidden Set Size: ", len(chosen_verts), " in ", len(points), " samples")
elif APPROACH ==4:
	def sample_node(w):
		return w.sample_cfree(1)[0]

	sample_node_handle = partial(sample_node, w = world)
	dg = DoubleGreedyPartialVisbilityGraph(alpha = 0.001,
					eps = 0.001,
					max_samples = n,
					sample_node_handle=sample_node_handle,
					los_handle = world.visible,
					verbose=True)
	chosen_verts = np.array(dg.construct_independent_set())
	chosen_verts = np.array(dg.refine_independent_set_greedy())
	ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
	v_sampleset = np.array([dg.sample_set[k][0] for k in dg.sample_set.keys()])
	ax.scatter(v_sampleset[:,0], v_sampleset[:,1], color="k", s = 1)
	print("Double Greedy Solution")
	print("Hidden Set Size: ", len(chosen_verts), " in ", len(dg.points), " samples")
else:
	raise NotImplementedError("Choose valid approach {1,2,3,4}, chosen", APPROACH)
regions, seed_points = generate_regions_multi_threading(chosen_verts, world.obstacle_triangles, world.iris_domain)
print('done generating regions')
for r in regions:
	world.plot_HPoly(ax, r)

plt.draw()


plt.show()
plt.waitforbuttonpress()
print('done')