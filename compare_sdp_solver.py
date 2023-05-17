import numpy as np
from visibility_graphs import get_visibility_graph
from independent_set import solve_max_independent_set_binary_quad_GW
import matplotlib.pyplot as plt
from cgdataset import World
from pydrake.all import MosekSolver, ScsSolver

world_name = "srpg_iso_aligned_mc0000172.instance.json"
world = World("./data/examples_01/" + world_name)
n = 450
seed = 0

points, adj_mat, _, _, _ = get_visibility_graph(world_name, world, n, seed)
adj_mat = adj_mat.toarray()

num = 300

solver_names = ["Mosek", "SCS"]
solvers = [MosekSolver(), ScsSolver()]
for solver_name, solver in zip(solver_names, solvers):
	count, _ = solve_max_independent_set_binary_quad_GW(adj_mat[:num,:num], sdp_solver=solver)
	print("%s: %d" % (solver_name, count))