import numpy as np
from independent_set_solver import solve_max_independent_set_integer
from seeding_utils import shrink_regions
import time
from time import strftime,gmtime

class VisSeeder:
    def __init__(self,
                 N = 400, #computational cost per iteration
                 alpha = 0.05, #bernoulli test parameter 1
                 eps = 0.05, #bernoulli test parameter 2
                 max_iterations = 10,
                 sample_cfree = None,
                 build_vgraph = None,
                 iris_w_obstacles = None,
                 verbose = False
                 ):
        
        self.vb = verbose
        self.sample_cfree = sample_cfree
        self.build_vgraph = build_vgraph
        self.iris_w_obstacles = iris_w_obstacles
        self.N = N
        self.alpha = alpha
        self.eps = eps
        self.maxit = max_iterations
        self.M = int(np.log(alpha)/np.log(1-eps))
        if self.vb: 
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VisSeeder] GuardInsertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +f"[VisSeeder] {1-self.alpha:.2f} probability that unseen region is less than {100*eps:.1f} '%' of Cfree ")
        
        self.vgraph_points = []
        self.vgraph_admat = []
        self.seed_points = []
        self.regions = []

    def run(self):
        done = False
        it = 0
        while it<self.maxit:

            #sample N points in cfree
            points, b_test_is_full = self.sample_cfree(self.N, self.M, self.regions)
            self.vgraph_points.append(points)
            if b_test_is_full:
                if self.vb : print(strftime("[%H:%M:%S] ", gmtime()) +'[VisSeeder] Bernoulli test failed')
                done = True 
                break

            #build visibility graph
            ad_mat = self.build_vgraph(points, self.regions)
            self.vgraph_admat.append(ad_mat)

            #solve MHS
            _, mhs_idx = solve_max_independent_set_integer(ad_mat)
            self.seed_points +=[points[mhs_idx, :]]
            if self.vb : print(strftime("[%H:%M:%S] ", gmtime()) +'[VisSeeder] Found ', len(mhs_idx), ' hidden points')

            #grow the regions with obstacles
            regions_step = self.iris_w_obstacles(points[mhs_idx, :], shrink_regions(self.regions, offset_fraction=0.25))
            self.regions += regions_step

        return self.regions
    