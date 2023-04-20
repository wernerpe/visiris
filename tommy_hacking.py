import numpy as np
from tqdm import tqdm
from functools import partial
from cgdataset import World
from independent_set import solve_lovasz_sdp, solve_max_independent_set_integer, solve_max_independent_set_binary_quad_GW, DoubleGreedy
from pydrake.geometry.optimization import (
    HPolyhedron, VPolytope, Iris, IrisOptions, Hyperellipsoid)
import time

iris_options = IrisOptions()
iris_options.require_sample_point_is_contained = True
iris_options.iteration_limit = 5
iris_options.termination_threshold = -1
iris_options.relative_termination_threshold = 0.05

def generate_regions(pts, iris_handle):
	regions = []
	succ_seed_pts = []
	for idx, pt in enumerate(pts):
		print(time.strftime("[%H:%M:%S] ", time.gmtime()), idx+1, '/', len(pts))
		try:
			reg = iris_handle(pt.reshape(-1,1))
			regions.append(reg)
			succ_seed_pts.append(pt)
		except:
			print('Iris failed at ', pt)
	return regions, succ_seed_pts

def iris_mut_env(seed, obstacles, domain, iris_opt):
    return Iris(obstacles, seed, domain, iris_opt)


favorite_polys = ["srpg_iso_aligned_mc0000172.instance.json", "cheese102.instance.json", "srpg_mc0000579.instance.json", "fpg-poly_0000000070_h1.instance.json"]

world = World("./data/examples_01/"+favorite_polys[-1])
n = 200

np.random.seed(0)
iris_handle = partial(iris_mut_env, obstacles = world.obstacle_triangles, domain = world.iris_domain, iris_opt = iris_options)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
world.plot_cfree(ax)
world.plot_triangles(ax)
plt.draw()
plt.pause(0.001)

points = np.zeros((n,2))
adj_mat = np.zeros((n,n))
for i in tqdm(range(n)):
	point = world.sample_cfree(1)[0]
	#ax.scatter([point[0]], [point[1]], color="black")
	for j in range(len(points)):
		other = points[j]
		if world.visible(point, other):
			#ax.plot([point[0], other[0]], [point[1], other[1]], color="black", linewidth=0.25, alpha = 0.5)
			adj_mat[i,j] = adj_mat[j,i] = 1
	points[i] = point
plt.draw()
plt.pause(0.001)

#theta, mat = solve_lovasz_sdp(adj_mat)
#print("Lovasz ")
#print(theta)
#print(mat)

# m, verts = solve_max_independent_set_integer(adj_mat)
# print("Integer Solution")
# print('Independent set sol', m)
#print(verts)
# chosen_verts = points[np.nonzero(verts)]
# ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
# plt.draw()
# plt.waitforbuttonpress()
#ax.scatter(points[:,0], points[:,1], color="black")

def sample_node(w):
	return w.sample_cfree(1)[0]

sample_node_handle = partial(sample_node, w = world)
dg = DoubleGreedy(alpha = 0.01,
		  		  eps = 0.001,
				  max_samples = 1500,
				  sample_node_handle=sample_node_handle,
				  los_handle = world.visible,
				  verbose=True)
chosen_verts = np.array(dg.construct_independent_set())
chosen_verts = np.array(dg.refine_independent_set_greedy())
ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="blue")
v_sampleset = np.array([dg.sample_set[k][0] for k in dg.sample_set.keys()])
ax.scatter(v_sampleset[:,0], v_sampleset[:,1], color="k", s = 1)

print("Hidden Set Size: ", len(chosen_verts), " in ", len(dg.points), " samples")
regions, seed_points = generate_regions(chosen_verts, iris_handle)
print('done generating regions')
for r in regions:
	world.plot_HPoly(ax, r)

plt.draw()
plt.waitforbuttonpress()

# m, verts = solve_max_independent_set_binary_quad_GW(adj_mat, n_rounds=1000)
# print('Binary Quad Relaxation + Rounding', m)
# #print(verts)
# chosen_verts = points[np.nonzero(verts)]
# ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
plt.draw()
plt.show()
print('done')