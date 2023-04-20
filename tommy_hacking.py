import numpy as np
from tqdm import tqdm
from functools import partial
from cgdataset import World
from independent_set import solve_lovasz_sdp, solve_max_independent_set_integer, solve_max_independent_set_binary_quad_GW, DoubleGreedy

world = World("./data/examples_01/srpg_iso_aligned_mc0000172.instance.json")
n = 100

np.random.seed(0)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
world.plot_cfree(ax)
plt.draw()
plt.pause(0.001)

points = np.zeros((n,2))
adj_mat = np.zeros((n,n))
for i in tqdm(range(n)):
	point = world.sample_cfree(1)[0]
	ax.scatter([point[0]], [point[1]], color="black")
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

m, verts = solve_max_independent_set_integer(adj_mat)
print("Integer Solution")
print('Independent set sol', m)
#print(verts)
chosen_verts = points[np.nonzero(verts)]
ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
plt.draw()
plt.waitforbuttonpress()
#ax.scatter(points[:,0], points[:,1], color="black")

def sample_node(w):
	return w.sample_cfree(1)[0]

sample_node_handle = partial(sample_node, w = world)
dg = DoubleGreedy(alpha = 0.01,
		  		  eps = 0.005,
				  max_samples = 1500,
				  sample_node_handle=sample_node_handle,
				  los_handle = world.visible,
				  verbose=True)
chosen_verts = np.array(dg.construct_independent_set())
chosen_verts = np.array(dg.refine_independent_set_greedy())
ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="blue")
print("Hidden Set Size: ", len(chosen_verts), " in ", len(dg.points), " samples")
plt.draw()
plt.waitforbuttonpress()

m, verts = solve_max_independent_set_binary_quad_GW(adj_mat, n_rounds=1000)
print('Binary Quad Relaxation + Rounding', m)
#print(verts)
chosen_verts = points[np.nonzero(verts)]
ax.scatter(chosen_verts[:,0], chosen_verts[:,1], color="red")
plt.draw()
plt.show()