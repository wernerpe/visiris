import numpy as np
from pydrake.all import MathematicalProgram, Solve

def graph_complement(adj_mat):
	return (1 - adj_mat) - np.eye(adj_mat.shape[0])

def solve_lovasz_sdp(adj_mat):
	n = adj_mat.shape[0]
	prog = MathematicalProgram()
	B = prog.NewSymmetricContinuousVariables(n)
	J = np.ones((n,n))

	prog.AddPositiveSemidefiniteConstraint(B)
	prog.AddLinearConstraint(np.trace(B) == 1)
	for i in range(0,n):
		for j in range(i,n):
			if adj_mat[i,j]:
				prog.AddLinearConstraint(B[i,j] == 0)
	prog.AddLinearCost(-np.trace(B @ J))

	result = Solve(prog)
	return -result.get_optimal_cost(), result.GetSolution(B)

if __name__ == "__main__":
	graph = np.array([
		[0, 1, 0],
		[1, 0, 1],
		[0, 1, 0]
	])
	size, mat = solve_lovasz_sdp(graph)
	print(size)
	print(mat)