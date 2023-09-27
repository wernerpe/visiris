import numpy as np
from visibility_graphs import get_visibility_graph
from old_scripts.independent_set import solve_max_independent_set_binary_quad_GW
import matplotlib.pyplot as plt
from cgdataset import World
from pydrake.all import MosekSolver, ScsSolver
import resource

world_name = "srpg_iso_aligned_mc0000172.instance.json"
world = World("./data/examples_01/" + world_name)
n = 450
seed = 0

points, adj_mat, _, _, _ = get_visibility_graph(world_name, world, n, seed)
adj_mat = adj_mat.toarray()

num = 200

solver_names = ["SCS", "MOSEK"]
solvers = [ScsSolver(), MosekSolver()]
for solver_name, solver in zip(solver_names, solvers):
	count, _ = solve_max_independent_set_binary_quad_GW(adj_mat[:num,:num], sdp_solver=solver)
	print("%s: %d" % (solver_name, count))
	print("Memory usage: %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	# This is the overall maximum, but we know SCS will use less, so this will work