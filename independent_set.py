import numpy as np
from tqdm import tqdm
from pydrake.all import MathematicalProgram, Solve, SolverOptions, CommonSolverOption

def solve_lovasz_sdp(adj_mat):
	print("Setting Up Mathematical Program")
	n = adj_mat.shape[0]
	prog = MathematicalProgram()
	B = prog.NewSymmetricContinuousVariables(n)
	J = np.ones((n,n))

	prog.AddPositiveSemidefiniteConstraint(B)
	prog.AddLinearConstraint(np.trace(B) == 1)
	print("Adding Graph Constraints")
	for i in tqdm(range(0,n)):
		for j in range(i,n):
			if adj_mat[i,j]:
				prog.AddLinearConstraint(B[i,j] == 0)
	print("Adding Cost")
	# prog.AddLinearCost(-np.trace(B @ J))
	prog.AddLinearCost(-np.sum(np.multiply(B, J)))
	print("Done!")

	solver_options = SolverOptions()
	solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

	result = Solve(prog, solver_options=solver_options)
	return -result.get_optimal_cost(), result.GetSolution(B)

def solve_max_independent_set_integer(adj_mat):
	n = adj_mat.shape[0]
	prog = MathematicalProgram()
	v = prog.NewBinaryVariables(n)
	prog.AddLinearCost(-np.sum(v))
	for i in range(0,n):
		for j in range(i,n):
			if adj_mat[i,j]:
				prog.AddLinearConstraint(v[i] + v[j] <= 1)


	solver_options = SolverOptions()
	solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

	result = Solve(prog, solver_options=solver_options)
	return -result.get_optimal_cost(), result.GetSolution(v)

if __name__ == "__main__":
	graph = np.array([
		[0, 1, 0],
		[1, 0, 1],
		[0, 1, 0]
	])
	size, mat = solve_lovasz_sdp(graph)
	print(size)
	print(mat)