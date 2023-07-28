from cgdataset import World
import matplotlib.pyplot as plt
import numpy as np
from seeding_utils import point_in_regions, point_near_regions, vis_reg, compute_kernels
#from independent_set_solver import solve_max_independent_set_integer
from visibility_clique_decomposition import VisCliqueDecomp
from scipy.sparse import lil_matrix
from tqdm import tqdm
from pydrake.all import Hyperellipsoid, VPolytope
from vislogging import CliqueApproachLogger
from region_generation import generate_regions_ellipses_multi_threading
from seeding_utils import sorted_vertices
import shapely
from shapely.ops import cascaded_union
from time import strftime, gmtime

eps_sample = -0.05
eps = 0.1
N = 500
#seed = 5

for world_name in ["cheese102.instance.json"]:#"fpg-poly_0000000300_h5.instance.json","srpg_iso_aligned_mc0000172.instance.json","fpg-poly_0000000060_h1.instance.json","fpg-poly_0000000070_h1.instance.json"]:
      for seed in [1,2,3]:
                    
        np.random.seed(seed)
        #world_name = "cheese102.instance.json"#small_polys[1] #"cheese205.instance.json"#fpg-poly_0000000060_h1.instance.json"#"srpg_iso_aligned_mc0000172.instance.json"##"fpg-poly_0000000070_h1.instance.json"
        world = World("./data/evalexamples/"+world_name, seed=seed)
        world.build_offset_cfree(eps_sample)

        def sample_cfree_handle(n, m, regions=None):
            points = np.zeros((n,2))
            if regions is None: regions = []		
            for i in range(n):
                bt_tries = 0
                while bt_tries<m:
                    point = world.sample_cfree_distance(1, eps = eps_sample)[0]
                    #point = world.sample_cfree(1)[0]
                    if point_near_regions(point, regions, tries = 100, eps = 0.1):
                        bt_tries+=1
                    else:
                        break
                if bt_tries == m:
                    return points, True
                
                points[i] = point
            return points, False

        def vgraph_builder(points):
            n = len(points)
            adj_mat = lil_matrix((n,n))
            for i in tqdm(range(n)):
                point = points[i, :]
                for j in range(len(points[:i])):
                    other = points[j]
                    # if region_vis_obstacles:
                    # 	if vis_reg(point, other, world, []):
                    # 		adj_mat[i,j] = adj_mat[j,i] = 1
                    # else:
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

        def col_handle(pt):
            return world.visible(pt,pt)

        def iris_w_obstacles_handle(points, ellipsoids, old_regions = None):
            if N>=1:
                #+ region_obstacles
                obstacles = [r for r in world.obstacle_triangles]
                regions, _, is_full = generate_regions_ellipses_multi_threading(points, ellipsoids, obstacles, world.iris_domain, compute_coverage, coverage_threshold=1-eps, old_regs = old_regions, maxiters=3)
            # else:
            #     #if N=1 coverage estimate happens at every step
            #     obstacles = [r for r in world.obstacle_triangles]
            #     if use_region_obstacles:
            #         obstacles += region_obstacles
            #     regions, _, _ = generate_regions_ellipses_multi_threading(points, ellipsoids, obstacles, world.iris_domain)
            #     is_full = 1-eps <= compute_coverage(old_regions+regions)
            return regions, is_full

        loggerccd = CliqueApproachLogger(world, world_name,"_nxccv_" +strftime("_%m%d%H%M%S_", gmtime()), seed, N, eps, plt_time = 10)

        VCD = VisCliqueDecomp(  N = N,
                                eps = eps,
                                max_iterations=10,
                                sample_cfree=sample_cfree_handle,
                                col_handle=col_handle,
                                build_vgraph=vgraph_builder,
                                iris_w_obstacles= iris_w_obstacles_handle,
                                verbose=True,
                                logger=loggerccd
                            )

        VCD.run()

        print('done')