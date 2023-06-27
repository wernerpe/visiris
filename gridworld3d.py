from gridenv import GridWorld3D
from PIL import Image
import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw
from seeding_utils import point_near_regions, vis_reg, point_in_regions
from region_generation import generate_regions_multi_threading3D
from tqdm import tqdm
from scipy.sparse import lil_matrix


side_len = 5
seed = 2
#X_WC = RigidTransform(RollPitchYaw(0,0,0),np.array(3*[2*side_len]) ) # some drake.RigidTransform()
#world.meshcat.SetTransform("/Cameras/default", X_WC) 


alpha = 0.05
eps = 0.1
for b in [5]:
    for seed in [1,2,3]:
        for N in [1, 30, 300]:

            world = GridWorld3D(b, side_len, seed=seed)
            def sample_cfree_handle(n, m, regions=None):
                points = np.zeros((n,3))
                if regions is None: regions = []		
                for i in range(n):
                    bt_tries = 0
                    while bt_tries<m:
                        point = world.sample_cfree(1)[0]
                        if point_near_regions(point, regions, eps = 0.1):
                            bt_tries+=1
                        else:
                            break
                    if bt_tries == m:
                        return points, True
                    points[i] = point
                return points, False

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

            def iris_w_obstacles(points, region_obstacles):
                regions, _ = generate_regions_multi_threading3D(points, world.obstacels + region_obstacles, world.iris_domain)
                return regions

            def estimate_coverage(regions):
                n_s = 1000
                samples = world.sample_cfree(n_s)
                in_s = 0
                for s in samples:
                    if point_in_regions(s, regions):
                        in_s+=1
                return (1.0*in_s)/n_s


            def iris_w_obstacles(points, region_obstacles, old_regions = None):
                if N>1:
                    regions, _, is_full = generate_regions_multi_threading3D(points, world.obstacles + region_obstacles, world.iris_domain, estimate_coverage, coverage_threshold=1-eps, old_regs = old_regions)
                else:
                    #if N=1 coverage estimate happens at every step
                    regions, _, is_full = generate_regions_multi_threading3D(points, world.obstacles + region_obstacles, world.iris_domain)
                return regions, is_full


            from vislogging import Logger3D
            from visibility_seeding import VisSeeder
            # # Capture the image
            # image_data = w.meshcat.StaticHtml()
            logger = Logger3D(world, f"grid3d_{b}", seed, N, alpha, eps, estimate_coverage)
            VS = VisSeeder(N = N,
                        alpha = alpha,
                        eps = eps,
                        max_iterations = 300,
                        sample_cfree = sample_cfree_handle,
                        build_vgraph = vgraph_builder,
                        iris_w_obstacles = iris_w_obstacles,
                        verbose = True,
                        logger = logger
                        )

            regions = VS.run()

#world.plot_regions(regions)

# with open()
# import imgkit
# def save_html_as_png(html_code, output_file):
#     #config = imgkit.config(wkhtmltoimage='/path/to/wkhtmltoimage')
#     imgkit.from_string(html_code, output_file, options={'format': 'png'})

# save_html_as_png(image_data, 'test.png')

print('d')