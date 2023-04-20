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

def solve_max_independent_set_binary_quad_GW(adj_mat, n_rounds=100, n_constraint_fixes = 10):
	n = adj_mat.shape[0]
	prog = MathematicalProgram()
	V = prog.NewSymmetricContinuousVariables(n+1)
	Q = np.zeros((n+1,n+1))
	Q[0, 1:] =1
	Q[1:, 0] =1
	
	prog.AddLinearCost(-0.5*(n+0.5*np.trace(np.matmul(Q,V))))
	prog.AddPositiveSemidefiniteConstraint(V)
	for i in range(0,n):
		for j in range(i,n):
			if adj_mat[i,j]:
				prog.AddLinearConstraint(V[0, 0]+ V[0, i+1] + V[0, j+1] +V[i+1, j+1]==0)
				prog.AddLinearConstraint(V[0, 0]+ V[i+1, 0] + V[j+1, 0] +V[j+1, i+1]==0)
	
	for i in range(0,n+1):
		prog.AddLinearConstraint(V[i,i]==1)

	solver_options = SolverOptions()
	solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

	result = Solve(prog, solver_options=solver_options)

	C = np.linalg.cholesky(result.GetSolution(V)+ np.eye(n+1)*1e-8)
	U = C.T
	r = U.shape[0]
	vals_gw = []
	xsols = []

	for idx in range(n_rounds):
		#random hyperplane -> projecting a spherical gaussian to unit sphere
		a = np.random.randn(r)
		a /= np.linalg.norm(a)
		xsol = np.array([np.sign(a.T@U[:,i]) for i in range(n+1)])
		#resolve constraint violations
		sign_u0 = xsol[0]
		violating_nodes = {}
		for i in range(0,n):
			for j in range(i,n):
				if adj_mat[i,j]:
					if sign_u0 == xsol[i+1] and xsol[j+1] == sign_u0:
						violating_nodes[i] =  np.abs(a.T@U[:,j+1])
						violating_nodes[j] =  np.abs(a.T@U[:,j+1])

		for _ in range(1):#n_constraint_fixes):
			constrained_nodes = []
			vals = list(violating_nodes.values())
			nodes = list(violating_nodes.keys())
			order = np.argsort(vals)[::-1] #np.random.permutation(len(vals))
			fixed_sols = []
			fixed_vals = []
			x_tmp = xsol.copy()
			for node_to_fix in order:
				node = nodes[node_to_fix]
				ad = np.where(adj_mat[node, :] ==1)[0]
				if node not in constrained_nodes:
					constrained_nodes.append(node)
					x_tmp[node+1] = sign_u0
					for a in ad:
						x_tmp[a+1] = -sign_u0
			fixed_vals.append(0.5*(n+0.5*x_tmp.T@Q@x_tmp))
			fixed_sols.append(x_tmp)

		idxmax = np.argmax(fixed_vals)
		vals_gw.append(fixed_vals[idxmax])
		xsols.append(fixed_sols[idxmax])

	# for val, x in zip(vals_gw, xsols):
	# 	V_sol = x.reshape(-1,1)@x.reshape(1,-1)
	# 	violations = 0
	# 	for i in range(0,n):
	# 		for j in range(i,n):
	# 			if adj_mat[i,j]:
	# 				if V_sol[0, 0]+ V_sol[0, i+1] + V_sol[0, j+1] +V_sol[i+1, j+1]:
	# 					violations +=1
	# 				if V_sol[0, 0]+ V_sol[i+1, 0] + V_sol[j+1, 0] +V_sol[j+1, i+1]:
	# 					violations +=1
		#print('value: ', val, ' violations', violations)

	idxmax = np.argmax(vals_gw)
	xsol = xsols[idxmax]
	print('Relaxation: ', 0.5*(n+0.5*np.trace(np.matmul(Q, result.GetSolution(V)))), ' Rounding: ', vals_gw[idxmax])
	return vals_gw[idxmax], xsol[1:] == xsol[0]

if __name__ == "__main__":
	graph = np.array([
		[0, 1, 0],
		[1, 0, 1],
		[0, 1, 0]
	])
	size, mat = solve_lovasz_sdp(graph)
	print(size)
	print(mat)