import time
import multiprocessing as mp
from functools import partial
from pydrake.geometry.optimization import (
    HPolyhedron, VPolytope, Iris, IrisOptions, Hyperellipsoid)
import numpy as np

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

def generate_regions_multi_threading(pts, obstacles, domain):
	regions = []
	succ_seed_pts = []
	A_obs = [r.A() for r in obstacles]
	b_obs = [r.b() for r in obstacles]
	A_dom = domain.A()
	b_dom = domain.b()
	chunks = np.array_split(pts, mp.cpu_count())
	pool = mp.Pool(processes=mp.cpu_count())
	genreg_hand = partial(generate_regions,
		          		  Aobs = A_obs,
					      Bobs = b_obs,
					      Adom = A_dom,
					      bdom = b_dom
						  )
	results = pool.map(genreg_hand, chunks)
	for r in results:
		regions += r[0]
		succ_seed_pts += r[1]
	return regions, succ_seed_pts