from gridenv import GridWorld
import matplotlib.pyplot as plt
from seeding_utils import point_in_regions, vis_reg, sorted_vertices
from pydrake.all import VPolytope
from region_generation import generate_regions_multi_threading
from visibility_seeding import VisSeeder
from vislogging import Logger
from scipy.sparse import lil_matrix
import numpy as np
from tqdm import tqdm
import shapely
from shapely.ops import cascaded_union

# seed = 1
size = 5
# boxes = 9
alpha = 0.05
eps = 0.1
# N = 300
for boxes in [9,11]:
    for seed in [6,7]:
        for N in [1, 30, 300]:#[1, 30, 300]:
            if boxes == 9 and seed == 6:
                break
            world = GridWorld(boxes, side_len=size, seed = seed)
            # fig,ax = plt.subplots(figsize = (10,10))
            # ax.set_xlim((-size, size))
            # ax.set_ylim((-size, size))
            # world.plot_cfree(ax)
            # plt.pause(0.01)
            def sample_cfree_handle(n, m, regions=None):
                points = np.zeros((n,2))
                if regions is None: regions = []		
                for i in range(n):
                    bt_tries = 0
                    while bt_tries<m:
                        point = world.sample_cfree(1)[0]
                        in_col = False
                        for _ in range(10):
                            r = point + 0.2*(np.random.rand(2)-0.5)
                            #was bug in initial run (forgot +r)
                            if point_in_regions(point+r, regions):
                                in_col = True
                                bt_tries += 1
                                break
                        if bt_tries == m:
                            return points, True 
                        else:
                            if not in_col:
                                break
                    points[i] = point
                return points, False
            
            # def sample_cfree_handle(n, m, regions=None):
            #     points = np.zeros((n,2))
            #     if regions is None: regions = []		
            #     for i in range(n):
            #         bt_tries = 0
            #         while bt_tries<m:
            #             point = world.sample_cfree(1)[0]
            #             for _ in range(10):
            #                 r = 0.05(np.random.rand(2)-0.5)
            #             if point_in_regions(point, regions):
            #                 bt_tries += 1
            #                 if bt_tries == m:
            #                     return points, True 
            #             else:
            #                 break
            #         points[i] = point
            #     return points, False

            # pts, _ = sample_cfree_handle(100, 50, None)
            # ax.scatter(pts[:,0], pts[:,1])
            # plt.show()

            def vgraph_builder(points, regions):
                n = len(points)
                adj_mat = lil_matrix((n,n))
                for i in tqdm(range(n)):
                    point = points[i, :]
                    for j in range(len(points[:i])):
                        other = points[j]
                        if vis_reg(point, other, world, regions):
                            adj_mat[i,j] = adj_mat[j,i] = 1
                return adj_mat.toarray()

            # pts, _ = sample_cfree_handle(100, 50, None)
            # ad = vgraph_builder(pts, [])
            # id1, id2 = np.where(ad==1)
            # edges = []
            # for i1, i2 in zip(id1, id2):
            #     edges.append([[pts[i1,0], pts[i2,0]], [pts[i1,1], pts[i2,1]]])
            # ax.scatter(pts[:,0], pts[:,1])
            # for e in edges:
            #     ax.plot(e[0], e[1], linewidth = 0.5, c = 'k')

            # plt.show()
            def compute_coverage(regions):
                shapely_regions = []
                for r in regions:
                    verts = sorted_vertices(VPolytope(r))
                    shapely_regions.append(shapely.Polygon(verts.T))
                union_of_Polyhedra = cascaded_union(shapely_regions)
                return union_of_Polyhedra.area/world.cfree_polygon.area

            def iris_w_obstacles(points, region_obstacles, old_regions = None):
                if N>1:
                    regions, _, is_full = generate_regions_multi_threading(points, world.obstacles + region_obstacles, world.iris_domain, compute_coverage, coverage_threshold=1-eps, old_regs = old_regions)
                else:
                    #if N=1 coverage estimate happens at every step
                    regions, _, is_full = generate_regions_multi_threading(points, world.obstacles + region_obstacles, world.iris_domain)
                return regions, is_full

            logger = Logger(world, f"gridworld_single_{boxes}", seed, N, alpha, eps, plt_time= 20)
            VS = VisSeeder(N = N,
                        alpha = alpha,
                        eps = eps,
                        max_iterations = 400,
                        sample_cfree = sample_cfree_handle,
                        build_vgraph = vgraph_builder,
                        iris_w_obstacles = iris_w_obstacles,
                        verbose = True,
                        logger = logger
                        )

            regions = VS.run()
            