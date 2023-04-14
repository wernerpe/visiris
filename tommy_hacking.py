import numpy as np

from cgdataset import World
from independent_set import solve_lovasz_sdp, graph_complement

world = World("./data/examples_01/srpg_iso_aligned_mc0000172.instance.json")
n = 50

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
world.plot_cfree(ax)
plt.draw()
plt.pause(0.001)

points = np.zeros((n,2))
adj_mat = np.zeros((n,n))
for i in range(n):
	point = world.sample_cfree(1)[0]
	ax.scatter([point[0]], [point[1]], color="black")
	for j in range(len(points)):
		other = points[j]
		if world.visible(point, other):
			ax.plot([point[0], other[0]], [point[1], other[1]], color="black", linewidth=0.25)
			adj_mat[i,j] = adj_mat[j,i] = 1
	points[i] = point
	plt.draw()
	plt.pause(0.001)

theta, mat = solve_lovasz_sdp(adj_mat)
print(theta)
print(mat)

plt.show()