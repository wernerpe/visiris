import numpy as np
from visibility_graphs import get_visibility_graph
from old_scripts.independent_set import solve_max_independent_set_integer
import matplotlib.pyplot as plt
from cgdataset import World

world_name = "cheese132.instance.json"
world = World("./data/examples_01/" + world_name)
n = 1000
seed = 0

points, adj_mat, _, _, _ = get_visibility_graph(world_name, world, n, seed)

step = 100
set_size = []
for i in range(0, n+step, step):
	count, _ = solve_max_independent_set_integer(adj_mat[:i,:i])
	set_size.append(count)

plt.plot(np.arange(0, n+step, step), set_size)
plt.xlabel("Number of Points")
plt.ylabel("Maximum Independent Set Size")
plt.show()