import time
import multiprocessing as mp
from functools import partial
from pydrake.geometry.optimization import (
    HPolyhedron, VPolytope, Iris, IrisOptions, Hyperellipsoid)
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

def generate_regions(pts, Aobs, Bobs, Adom, bdom):
	iris_options = IrisOptions()
	iris_options.require_sample_point_is_contained = True
	iris_options.iteration_limit = 5
	iris_options.termination_threshold = -1
	iris_options.relative_termination_threshold = 0.05
	obstacles = [HPolyhedron(A, b) for A,b in zip(Aobs,Bobs)]
	domain = HPolyhedron(Adom, bdom)
	regions = []
	succ_seed_pts = []
	for idx, pt in enumerate(pts):
		print(time.strftime("[%H:%M:%S] ", time.gmtime()), idx+1, '/', len(pts))
		try:
			reg = Iris(obstacles, pt.reshape(-1,1), domain, iris_options)
			regions.append(reg)
			succ_seed_pts.append(pt)
		except:
			print('Iris failed at ', pt)
	return regions, succ_seed_pts

def generate_regions_ellipses(pts, Aobs, Bobs, Aell, cell, Adom, bdom, maxiters=5):
	assert len(pts)==len(Aell)
	iris_options = IrisOptions()
	iris_options.require_sample_point_is_contained = True
	iris_options.iteration_limit = maxiters
	iris_options.termination_threshold = -1
	iris_options.relative_termination_threshold = 0.05
	obstacles = [HPolyhedron(A, b) for A,b in zip(Aobs,Bobs)]
	domain = HPolyhedron(Adom, bdom)
	regions = []
	succ_seed_pts = []
	for idx, pt in enumerate(pts):
		print(time.strftime("[%H:%M:%S] ", time.gmtime()), idx+1, '/', len(pts))
		try:
			iris_options.initial_ellipsoid = Hyperellipsoid(Aell[idx],cell[idx])
			reg = Iris(obstacles, pt.reshape(-1,1), domain, iris_options)
			if np.array_equal(domain.A(), reg.A()):
				print('Iris failed at ', pt)
				return regions, succ_seed_pts
				#iris_options.initial_ellipsoid = Hyperellipsoid(Aell[idx],pt.reshape(-1,1))
				#reg = Iris(obstacles, pt.reshape(-1,1), domain, iris_options)
			regions.append(reg)
			succ_seed_pts.append(pt)
		except:
			print('Iris failed at ', pt)
	return regions, succ_seed_pts

#eliminates region obstacles for last n iterations
def generate_regions_regobs(pts, Aobs, Bobs, Aregobs, Bregobs, Adom, bdom, noregits =0):
	iris_options = IrisOptions()
	iris_options.require_sample_point_is_contained = True
	iris_options.iteration_limit = 5
	iris_options.termination_threshold = -1
	iris_options.relative_termination_threshold = 0.05

	iris_options_post = IrisOptions()
	iris_options_post.require_sample_point_is_contained = True
	iris_options_post.iteration_limit = noregits
	iris_options_post.termination_threshold = -1
	iris_options_post.relative_termination_threshold = 0.05

	obstacles = [HPolyhedron(A, b) for A,b in zip(Aobs,Bobs)]
	regobstacles = [HPolyhedron(A, b) for A,b in zip(Aregobs,Bregobs)]
	domain = HPolyhedron(Adom, bdom)
	regions = []
	succ_seed_pts = []
	for idx, pt in enumerate(pts):
		print(time.strftime("[%H:%M:%S] ", time.gmtime()), idx+1, '/', len(pts))
		try:
			reg_init = Iris(obstacles+regobstacles, pt.reshape(-1,1), domain, iris_options)
			if noregits ==0:
				print('skip')
				regions.append(reg_init)
				succ_seed_pts.append(pt)
			else:
				E = reg_init.MaximumVolumeInscribedEllipsoid()
				#shrink ellipsoid around center to avoid bad numerics aorund convergence
				Eshr = Hyperellipsoid(E.A()*10, E.center())
				iris_options_post.initial_ellipsoid = Eshr
				reg = Iris(obstacles, pt.reshape(-1,1), domain, iris_options_post)
				if reg.A().shape[0] == domain.A().shape[0]:
					#point leaves region on first iteration, catch error by replacing region with previous
					regions.append(reg_init)
					succ_seed_pts.append(pt)
				else:
					regions.append(reg)
					succ_seed_pts.append(pt)
		except:
			print('Iris failed at ', pt)
	return regions, succ_seed_pts

def generate_regions_multi_threading(pts, obstacles, domain, estimate_coverage = None, coverage_threshold = None, old_regs = None):
	is_full = False
	if len(pts.shape) == 2:
		dim = pts.shape[1]
	else:
		dim = len(pts)
	regions = []
	succ_seed_pts = []
	A_obs = [r.A() for r in obstacles]
	b_obs = [r.b() for r in obstacles]
	A_dom = domain.A()
	b_dom = domain.b()
	if len(pts.reshape(-1, dim))==1:
		regions, succ_seed_pts = generate_regions(pts.reshape(1,dim), A_obs, b_obs, A_dom, b_dom)
		cov = estimate_coverage(regions+old_regs)
		is_full = coverage_threshold<=cov
		return regions, succ_seed_pts, is_full
	
	genreg_hand = partial(generate_regions,
								Aobs = A_obs,
								Bobs = b_obs,
								Adom = A_dom,
								bdom = b_dom
								)
	#allow batched region generation to terminate early
	if coverage_threshold is not None and estimate_coverage is not None:
		nr_pts=len(pts)
		checks_per_n_regions = 1
		nr_checks = int(np.floor(nr_pts/checks_per_n_regions))
		chunk_list = []
		#split  into batches 
		for chunk_id in range(nr_checks):
			start_idx = checks_per_n_regions*chunk_id
			end_idx = checks_per_n_regions*(chunk_id+1)
			chunks = np.array_split(pts[start_idx:end_idx,:], mp.cpu_count()-5)
			chunk_list.append(chunks)
		chunk_list.append(np.array_split(pts[nr_checks*checks_per_n_regions:,:], mp.cpu_count()-5))
		
		for c in chunk_list:
			pool = mp.Pool(processes=mp.cpu_count()-5)
			results = pool.map(genreg_hand, c)
			for r in results:
				regions += r[0]
				succ_seed_pts += r[1]
			current_coverage_est = estimate_coverage(regions+old_regs)
			if current_coverage_est>= coverage_threshold:
				#space is already full, stop generating more regions
				return regions, succ_seed_pts, True
			
		return regions, succ_seed_pts, is_full

	else:
		chunks = np.array_split(pts, mp.cpu_count()-5)
		pool = mp.Pool(processes=mp.cpu_count()-5)
		results = pool.map(genreg_hand, chunks)
		for r in results:
			regions += r[0]
			succ_seed_pts += r[1]
		return regions, succ_seed_pts, is_full
	
def generate_regions_ellipses_multi_threading(pts, seed_ellipses, obstacles, domain, estimate_coverage = None, coverage_threshold = None, old_regs = None, maxiters=5):
	is_full = False
	regions = []
	succ_seed_pts = []
	A_obs = [r.A() for r in obstacles]
	b_obs = [r.b() for r in obstacles]
	A_ell = [s.A() for s in seed_ellipses]
	c_ell = [s.center() for s in seed_ellipses]
	A_dom = domain.A()
	b_dom = domain.b()
	#if len(pts.reshape(-1,2))==1:
	for pt, Aell, cell in zip(pts, A_ell, c_ell):
		region, succ_seed_pt = generate_regions_ellipses(pt.reshape(1,-1), A_obs, b_obs, [Aell], [cell], A_dom, b_dom, maxiters)
		if len(region)>0:
			regions += region
			cov = estimate_coverage(regions+old_regs)
			is_full = coverage_threshold<=cov
			#regions.append(region)
			succ_seed_pts.append(succ_seed_pt)
			if is_full: 
				return regions, succ_seed_pts, is_full
	return regions, succ_seed_pts, is_full
	
	# genreg_hand = partial(generate_regions_ellipses,
	# 							Aobs = A_obs,
	# 							Bobs = b_obs,
	# 							A_ell = A_ell,
	# 							c_ell = c_ell,
	# 							Adom = A_dom,
	# 							bdom = b_dom
	# 							)
	# #allow batched region generation to terminate early
	# if coverage_threshold is not None and estimate_coverage is not None:
	# 	nr_pts=len(pts)
	# 	checks_per_n_regions = 1
	# 	nr_checks = int(np.floor(nr_pts/checks_per_n_regions))
	# 	chunk_list = []
	# 	#split  into batches 
	# 	for chunk_id in range(nr_checks):
	# 		start_idx = checks_per_n_regions*chunk_id
	# 		end_idx = checks_per_n_regions*(chunk_id+1)
	# 		chunks = np.array_split(pts[start_idx:end_idx,:], mp.cpu_count()-5)
	# 		chunk_list.append(chunks)
	# 	chunk_list.append(np.array_split(pts[nr_checks*checks_per_n_regions:,:], mp.cpu_count()-5))
		
	# 	for c in chunk_list:
	# 		pool = mp.Pool(processes=mp.cpu_count()-5)
	# 		results = pool.map(genreg_hand, c)
	# 		for r in results:
	# 			regions += r[0]
	# 			succ_seed_pts += r[1]
	# 		current_coverage_est = estimate_coverage(regions+old_regs)
	# 		if current_coverage_est>= coverage_threshold:
	# 			#space is already full, stop generating more regions
	# 			return regions, succ_seed_pts, True
			
	# 	return regions, succ_seed_pts, is_full

	# else:
	# 	chunks = np.array_split(pts, mp.cpu_count()-5)
	# 	pool = mp.Pool(processes=mp.cpu_count()-5)
	# 	results = pool.map(genreg_hand, chunks)
	# 	for r in results:
	# 		regions += r[0]
	# 		succ_seed_pts += r[1]
	# 	return regions, succ_seed_pts, is_full

def generate_regions_multi_threading_regobs(pts, obstacles, region_obstacles, domain, estimate_coverage = None, coverage_threshold = None, old_regs = None, noregits = 1):
	is_full = False
	regions = []
	succ_seed_pts = []
	A_obs = [r.A() for r in obstacles]
	b_obs = [r.b() for r in obstacles]
	A_obs_reg = [r.A() for r in region_obstacles]
	b_obs_reg = [r.b() for r in region_obstacles]
	A_dom = domain.A()
	b_dom = domain.b()
	if len(pts.reshape(-1,2))==1:
		regions, succ_seed_pts = generate_regions_regobs(pts.reshape(1,2), A_obs, b_obs, A_obs_reg, b_obs_reg, A_dom, b_dom, noregits=noregits)
		return regions, succ_seed_pts, is_full
	
	genreg_hand = partial(generate_regions_regobs,
								Aobs = A_obs,
								Bobs = b_obs,
								Aregobs= A_obs_reg,
								Bregobs= b_obs_reg,
								Adom = A_dom,
								bdom = b_dom,
								noregits=noregits
								)
	#allow batched region generation to terminate early
	if coverage_threshold is not None and estimate_coverage is not None:
		nr_pts=len(pts)
		checks_per_n_regions = 10
		nr_checks = int(np.floor(nr_pts/checks_per_n_regions))
		chunk_list = []
		#split  into batches 
		for chunk_id in range(nr_checks):
			start_idx = checks_per_n_regions*chunk_id
			end_idx = checks_per_n_regions*(chunk_id+1)
			chunks = np.array_split(pts[start_idx:end_idx,:], mp.cpu_count()-5)
			chunk_list.append(chunks)
		chunk_list.append(np.array_split(pts[nr_checks*checks_per_n_regions:,:], mp.cpu_count()-5))
		
		for c in chunk_list:
			pool = mp.Pool(processes=mp.cpu_count()-5)

			#genreg_hand(c)
			results = pool.map(genreg_hand, c)
			for r in results:
				regions += r[0]
				succ_seed_pts += r[1]
			current_coverage_est = estimate_coverage(regions+old_regs)
			if current_coverage_est>= coverage_threshold:
				#space is already full, stop generating more regions
				return regions, succ_seed_pts, True
			
		return regions, succ_seed_pts, is_full

	else:
		chunks = np.array_split(pts, mp.cpu_count()-5)
		pool = mp.Pool(processes=mp.cpu_count()-5)
		results = pool.map(genreg_hand, chunks)
		for r in results:
			regions += r[0]
			succ_seed_pts += r[1]
		return regions, succ_seed_pts, is_full

def build_region_obstacles(regions):
	if regions is not None:
		obstacles = []
		for r in regions:
			offset = 0.25*np.min(1/np.linalg.eig(r.MaximumVolumeInscribedEllipsoid().A())[0])
			rnew = HPolyhedron(r.A(), r.b()-offset)
			obstacles.append(rnew)
	return obstacles

def generate_region_with_region_obstacles(pt, regions, obstacles,  domain):
	iris_options = IrisOptions()
	iris_options.require_sample_point_is_contained = True
	iris_options.iteration_limit = 5
	iris_options.termination_threshold = -1
	iris_options.relative_termination_threshold = 0.05
	obstacles = obstacles + build_region_obstacles(regions)
	try:
		region = Iris(obstacles, pt.reshape(-1,1), domain, iris_options)
	except:
		print('Iris failed at ', pt)
	return region

def get_visible_connected_components(idx_pt, connected_components, adj_mat):
		nr_vis = 0
		for component in connected_components:
			adj_reduced = adj_mat[idx_pt, :]
			adj_reduced = adj_reduced[list(component)]
			if np.any(adj_reduced==1):
				nr_vis +=1
		return nr_vis

def fill_remaining_space(ax, points, chosen_verts, adj_mat, ind_regions, iris_with_region_obstacles):
	regions = [r for r in ind_regions]
	seed_points = chosen_verts.copy()
	seed_point_index = []
	for pt in chosen_verts:
		seed_point_index.append(np.where(np.all(pt==points, axis =1))[0][0])
	
	points_to_check = [idx for idx in range(points.shape[0])]
	
	# region connectivity graph
	connectivity_graph = nx.Graph()
	for idx in seed_point_index:
		connectivity_graph.add_node(idx)

	for idx1, sp_idx1 in enumerate(seed_point_index):
		for idx2, sp_idx2 in enumerate(seed_point_index):
			if idx1 != idx2:
				r1 = regions[idx1]
				r2 = regions[idx2]
				if r1.IntersectsWith(r2):
					connectivity_graph.add_edge(sp_idx1,sp_idx2)
		    
	to_del = []
	for idx_pt in points_to_check:
		if point_in_regions(regions, points[idx_pt, :]):
			to_del.append(idx_pt)

	for el in to_del:
		points_to_check.remove(el)

	# ax.scatter(points[points_to_check, 0], points[points_to_check, 1], c = 'g')
	# plt.draw()
	# plt.pause(0.01)
	while len(points_to_check):
		print("candidates remaining: ", len(points_to_check))
		num_visible_connected_components = []
		connected_components = nx.connected_components(connectivity_graph)

		connected_components = [list(component) for component in connected_components]
		# for ids , col in zip(connected_components, ['r','m', 'y', 'b']):
		# 	ax.scatter(points[ids, 0 ], points[ids, 1], s = 60, c = col)
		# ax.scatter(points[points_to_check[1], 0 ], points[points_to_check[1], 1], s = 120, c = 'K')
		# plt.draw()
		# plt.pause(0.01)
		for idx_pt in points_to_check:
			value = get_visible_connected_components(idx_pt, connected_components,adj_mat)
			# adj_red = adj_mat[:, points_to_check]
			# value = len(np.where(adj_red[idx_pt]==1)[0])																		
			num_visible_connected_components.append(value)

		idx_grow_region = points_to_check[np.argmax(num_visible_connected_components)]
		seed_point = points[idx_grow_region]
		seed_points = np.concatenate((seed_points, seed_point.reshape(1,-1)), axis = 0)
		seed_point_index.append(idx_grow_region)
		region = iris_with_region_obstacles(seed_point, regions)
		regions.append(region)

		to_del = []
		for idx_pt in points_to_check:
			if point_in_regions([region], points[idx_pt, :]):
				to_del.append(idx_pt)
		for el in to_del:
			points_to_check.remove(el)

		#update connectivity graph with new region
		for idx1, sp_idx1 in enumerate(seed_point_index[:-1]):
			r1 = regions[idx1]
			if region.IntersectsWith(r1):
				connectivity_graph.add_edge(sp_idx1, seed_point_index[-1])
	return regions, seed_points, seed_point_index
	
def point_in_regions(regions, pt):
	for r in regions:
		if r.PointInSet(pt.reshape(-1,1)):
			return True
	else:
		return False

def generate_regions_multi_threading3D(pts, obstacles, domain, estimate_coverage = None, coverage_threshold = None, old_regs = None):
	is_full = False
	regions = []
	succ_seed_pts = []
	A_obs = [r.A() for r in obstacles]
	b_obs = [r.b() for r in obstacles]
	A_dom = domain.A()
	b_dom = domain.b()
	if len(pts.reshape(-1,3))==1:
		regions, succ_seed_pts = generate_regions(pts.reshape(1,3), A_obs, b_obs, A_dom, b_dom)
		return regions, succ_seed_pts, is_full
	
	genreg_hand = partial(generate_regions,
								Aobs = A_obs,
								Bobs = b_obs,
								Adom = A_dom,
								bdom = b_dom
								)
	#allow batched region generation to terminate early
	if coverage_threshold is not None and estimate_coverage is not None:
		nr_pts=len(pts)
		checks_per_n_regions = 10
		nr_checks = int(np.floor(nr_pts/checks_per_n_regions))
		chunk_list = []
		#split  into batches 
		for chunk_id in range(nr_checks):
			start_idx = checks_per_n_regions*chunk_id
			end_idx = checks_per_n_regions*(chunk_id+1)
			chunks = np.array_split(pts[start_idx:end_idx,:], mp.cpu_count()-5)
			chunk_list.append(chunks)
		chunk_list.append(np.array_split(pts[nr_checks*checks_per_n_regions:,:], mp.cpu_count()-5))
		
		for c in chunk_list:
			pool = mp.Pool(processes=mp.cpu_count()-5)
			results = pool.map(genreg_hand, c)
			for r in results:
				regions += r[0]
				succ_seed_pts += r[1]
			current_coverage_est = estimate_coverage(regions+old_regs)
			if current_coverage_est>= coverage_threshold:
				#space is already full, stop generating more regions
				return regions, succ_seed_pts, True
			
		return regions, succ_seed_pts, is_full

	else:
		chunks = np.array_split(pts, mp.cpu_count()-5)
		pool = mp.Pool(processes=mp.cpu_count()-5)
		results = pool.map(genreg_hand, chunks)
		for r in results:
			regions += r[0]
			succ_seed_pts += r[1]
		return regions, succ_seed_pts, is_full