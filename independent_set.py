import numpy as np
from tqdm import tqdm
from pydrake.all import MathematicalProgram, Solve, SolverOptions, CommonSolverOption
import pydrake
import networkx as nx
from time import strftime, gmtime
import random

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
	
	prog.AddLinearCost(-0.5*(n+0.5*np.sum(np.multiply(Q,V))))
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
	
	print('Probability of constraint violation')
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

class DoubleGreedy:
	def __init__(self,
	      		 Vertices,
				 Adjacency_matrix,
		 		 verbose = False,
				 seed = 0
				 ):
		np.random.seed(seed)
		random.seed(seed)
		self.verbose = verbose	
		self.independent_set = []
		self.unvisited_nodes = [i for i in range(len(Vertices))]
		self.sample_set = []
		self.points = []
		self.Adj_mat = Adjacency_matrix
		self.Vertices = Vertices
		# if self.verbose: 
		# 	print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Point insertion attempts M:', str(self.M))
		# 	print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] {} probability that unseen region is less than {} "%" of Cfree '.format(1-self.alpha, 100*eps))

	def sample_node(self,):
		#find a
		idx_rnd = random.choice(self.unvisited_nodes) 
		return idx_rnd
	
	def construct_independent_set(self,):
		it = 0
		while len(self.unvisited_nodes):
			p = self.sample_node()
			visible_points = np.where(self.Adj_mat[p,:] == 1)[0]
			if any(v in self.independent_set for v in visible_points):
				self.sample_set.append(p)
			else:
				#greedily add to independent set
				self.independent_set.append(p)
			self.unvisited_nodes.remove(p)
			if self.verbose and (len(self.unvisited_nodes)%50) == 0: print(strftime("[%H:%M:%S] ", gmtime()) +"[DoubleGreedy]  #Unvisited Nodes = ", str(len(self.unvisited_nodes))) 
		return [self.Vertices[v] for v in self.independent_set]

	def compute_kernel_of_hidden_point(self, point_index):
		assert point_index in self.independent_set
		ker = []
		adj_p = np.where(self.Adj_mat[point_index, :]==1)[0]
		for s in self.sample_set:
			if s in adj_p:
				adj_s  = np.where(self.Adj_mat[s]==1)[0]
				if len(np.where([v in self.independent_set for v in adj_s])[0])==1:
					ker.append(s)
				else:
					pass
		return ker
	
	def get_new_set_candidates(self, point_index):
		kernel_points_index = self.compute_kernel_of_hidden_point(point_index)
		kernel_points_index += [point_index]
		if len(kernel_points_index)>1:
			reduced_adjacency_rows = self.Adj_mat[kernel_points_index]
			reduced_adjacency = reduced_adjacency_rows[:, kernel_points_index]
			rows, cols = np.where(reduced_adjacency == 1)
			edges = zip(rows.tolist(), cols.tolist())
			graph = nx.Graph()
			graph.add_edges_from(edges)
			idx_ind = nx.maximal_independent_set(graph)
			return [kernel_points_index[i] for i in idx_ind] 
		else:
			return [point_index]
		
	def refine_independent_set_greedy(self):
		continue_splitting = True
		while continue_splitting:
			candidate_splits = [self.get_new_set_candidates(p) for p in self.independent_set]
			best_split = max(candidate_splits, key = len)
			best_split_idx = candidate_splits.index(best_split)
			if len(best_split)>1:
				if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Hidden point found to split into', len(best_split))
				self.independent_set.remove(self.independent_set[best_split_idx])
				self.independent_set += best_split
			else:
				continue_splitting = False
		return [self.Vertices[ih] for ih in self.independent_set]
	


class DoubleGreedyPartialVisbilityGraph:
	def __init__(self,
	      		 alpha = 0.05,
                 eps = 0.05,
		 		 max_samples = 500,
		 		 sample_node_handle = None,
				 los_handle= None,
				 verbose = False
				 ):
		
		self.verbose = verbose
		self.alpha = alpha
		self.eps = eps
		self.max_samples = max_samples
		self.M = int(np.log(alpha)/np.log(1-eps))

		self.sample_node = sample_node_handle
		self.is_los = los_handle
		self.sample_set = {}
		self.hidden_set = []
		self.points = []
		if self.verbose: 
			print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Point insertion attempts M:', str(self.M))
			print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] {} probability that unseen region is less than {} "%" of Cfree '.format(1-self.alpha, 100*eps))

	def construct_independent_set(self,):
		it = 0
		while it < self.M:
			p = self.sample_node()
			self.points.append(p)
			visible_points = []
			add_to_sample_set = False
			for idx_point in self.hidden_set:
				hidden_point = self.points[idx_point]
				if self.is_los(p, hidden_point):
					add_to_sample_set = True
					visible_points.append(hidden_point)
			if add_to_sample_set:
				self.sample_set[str(p)] = [p, visible_points]
			else:
				if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +"[DoubleGreedy] New hidden point placed N = ", str(len(self.hidden_set)), "it = ", it) 
				self.hidden_set.append(len(self.points)-1)
				it = 0
				 #update visibility 
				for s_key in self.sample_set.keys():
					p_sample = self.sample_set[s_key][0]
					if self.is_los(p, p_sample):
						self.sample_set[s_key][1].append(p)
			it+=1
			if len(self.points)>=self.max_samples:
				if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Point budget exceeded', len(self.points))
				return  [self.points[ih] for ih in self.hidden_set]
		if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Sample set size',len(self.sample_set.keys()))

		return [self.points[ih] for ih in self.hidden_set]

	def compute_kernel_of_hidden_point(self, point):
		ker = []
		for sampdat in self.sample_set.values():
			pos = sampdat[0]
			if len(sampdat[1])==1 and np.array_equal(sampdat[1][0], self.points[point]):
				ker.append(pos)
		return ker
	
	def get_new_set_candidates(self, point):
		kernel_points = self.compute_kernel_of_hidden_point(point)
		kernel_points += [self.points[point]]
		graph = nx.Graph()
		if len(kernel_points)>1:
			for i1, v1 in enumerate(kernel_points):
				for i2, v2 in enumerate(kernel_points):
					if i1!=i2 and self.is_los(v1, v2):
						graph.add_edge(i1,i2)
			if len(graph.edges):
				new_cands = nx.maximal_independent_set(graph)
			else:
				raise ValueError("no edges detected, must be error")
			return [kernel_points[c] for c in new_cands]
		else:
			return [self.points[point]]

	def refine_independent_set_greedy(self):
		continue_splitting = True
		while continue_splitting:
			candidate_splits = [self.get_new_set_candidates(p) for p in self.hidden_set]
			best_split = max(candidate_splits, key = len)
			best_split_idx = candidate_splits.index(best_split)
			if len(best_split)>1:
				if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[DoubleGreedy] Hidden point found to split into', len(best_split))
				hidden_point_old = self.hidden_set[best_split_idx]
				p_hidden_old  = self.points[hidden_point_old]
				for s_key in self.sample_set.keys():
					vis_points = self.sample_set[s_key][1] 
					eq_list = [np.array_equal(v, p_hidden_old) for v in vis_points]
					idx_eq = eq_list.index(True) if True in eq_list else None
					if idx_eq is not None:
						vis_points.pop(idx_eq)
					p_sample = self.sample_set[s_key][0]
					for idnr, p_split in enumerate(best_split):
						if self.is_los(p_split, p_sample):
							vis_points.append(p_split)
				#remove old hidden point from hidden set
				self.hidden_set.pop(best_split_idx)
				for c in best_split:
					self.hidden_set.append([np.array_equal(v, c) for v in self.points].index(True))
			else:
				continue_splitting = False
		return [self.points[ih] for ih in self.hidden_set]
	
if __name__ == "__main__":
	graph = np.array([
		[0, 1, 0],
		[1, 0, 1],
		[0, 1, 0]
	])
	size, mat = solve_lovasz_sdp(graph)
	print(size)
	print(mat)