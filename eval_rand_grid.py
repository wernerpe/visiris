from gridenv import GridWorldRand
import matplotlib.pyplot as plt
from seeding_utils import point_near_regions, vis_reg, sorted_vertices
from pydrake.all import VPolytope
from region_generation import generate_regions_multi_threading
from visibility_seeding import VisSeeder
from vislogging import Logger
from scipy.sparse import lil_matrix
import numpy as np
from tqdm import tqdm
import shapely
from shapely.ops import cascaded_union

size = 5
# boxes = 9
alpha = 0.05
eps = 0.1
N = 1
eps_sample = 0.01
rand = 0.3
# N = 300
for boxes in [14]:
    for seed in [1,2,3]:
        world = GridWorldRand(boxes,
		       		  rand=rand, 
					  side_len=size, 
					  seed = seed, 
					  eps_offset=eps_sample)
        
        fig,ax = plt.subplots(figsize = (10,10))
        #ax.set_xlim((-size, size))
        #ax.set_ylim((-size, size))
        #world.plot_cfree_inflated(ax)
        #plt.pause(0.01)

        def sample_cfree_handle(n, m, regions=None):
            points = np.zeros((n,2))
            m = m*100
            if regions is None: regions = []		
            for i in range(n):
                bt_tries = 0
                while bt_tries<m:
                    point = world.sample_cfree_offset(1)[0]
                    #point = world.sample_cfree(1)[0]
                    if point_near_regions(point, regions, tries = 100, eps = 0.1):
                        bt_tries+=1
                    else:
                        break
                # if bt_tries == m:
                #     return points, True
                points[i] = point
            return points, False
            
        def vgraph_builder(points, regions, region_vis_obstacles=True):
            n = len(points)
            adj_mat = lil_matrix((n,n))
            for i in tqdm(range(n)):
                point = points[i, :]
                for j in range(len(points[:i])):
                    other = points[j]
                    if region_vis_obstacles:
                        if vis_reg(point, other, world, regions):
                            adj_mat[i,j] = adj_mat[j,i] = 1
                    else:
                        if vis_reg(point, other, world, []):
                            adj_mat[i,j] = adj_mat[j,i] = 1
            return adj_mat.toarray()

        def compute_coverage(regions):
            shapely_regions = []
            for r in regions:
                verts = sorted_vertices(VPolytope(r))
                shapely_regions.append(shapely.Polygon(verts.T))
            union_of_Polyhedra = cascaded_union(shapely_regions)
            return union_of_Polyhedra.area/world.cfree_polygon.area

        def iris_w_obstacles(points, region_obstacles, old_regions = None, use_region_obstacles = True):
            if N>1:
                #+ region_obstacles
                obstacles = [r for r in world.obstacles]
                if use_region_obstacles:
                    obstacles += region_obstacles
                regions, _, is_full = generate_regions_multi_threading(points, obstacles, world.iris_domain, compute_coverage, coverage_threshold=1-eps, old_regs = old_regions)
            else:
                #if N=1 coverage estimate happens at every step
                obstacles = [r for r in world.obstacles]
                if use_region_obstacles:
                    obstacles += region_obstacles
                regions, _, _ = generate_regions_multi_threading(points, obstacles, world.iris_domain, compute_coverage, coverage_threshold=1-eps, old_regs=old_regions)
                is_full = 1-eps <= compute_coverage(old_regions+regions)
            return regions, is_full

        logger = Logger(world, f"gridworld_rand_{boxes}",f"_{rand}", seed, N, alpha, eps, plt_time= 10)
        VS = VisSeeder(N = N,
                    alpha = alpha,
                    eps = eps,
                    max_iterations = 400,
                    sample_cfree = sample_cfree_handle,
                    build_vgraph = vgraph_builder,
                    iris_w_obstacles = iris_w_obstacles,
                    verbose = True,
                    logger = logger,
                    region_pullback= 0.25
                    )

        regions = VS.run()
        