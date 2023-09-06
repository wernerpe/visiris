from cgdataset import World
import matplotlib.pyplot as plt
import numpy as np
from seeding_utils import point_in_regions, point_near_regions, vis_reg, compute_kernels
from independent_set_solver import solve_max_independent_set_integer
from scipy.sparse import lil_matrix
from tqdm import tqdm
from pydrake.all import Hyperellipsoid

eps_sample = -0.05
world_name = "cheese102.instance.json"#small_polys[1] #"cheese205.instance.json"#fpg-poly_0000000060_h1.instance.json"#"srpg_iso_aligned_mc0000172.instance.json"##"fpg-poly_0000000070_h1.instance.json"
world = World("./data/evalexamples/"+world_name)
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


import numpy as np
from pydrake.all import (MathematicalProgram, SolverOptions, 
			            Solve, CommonSolverOption)

from ellipse_utils import get_lj_ellipse
import networkx as nx
def collision(pt):
    return world.visible(pt,pt)

def get_kernel_iris_metrics(cliques):
    seed_ellipses = [get_lj_ellipse(k) for k in cliques]
    seed_points = []
    for k,se in zip(cliques, seed_ellipses):
        center = se.center()
        if not collision(center):
            distances = np.linalg.norm(np.array(k).reshape(-1,2) - center, axis = 1).reshape(-1)
            mindist_idx = np.argmin(distances)
            seed_points.append(k[mindist_idx])
        else:
            seed_points.append(center)

    #rescale seed_ellipses
    mean_eig_scaling = 1000
    seed_ellipses_scaled = []
    for e in seed_ellipses:
        eigs, _ = np.linalg.eig(e.A())
        mean_eig_size = np.mean(eigs)
        seed_ellipses_scaled.append(Hyperellipsoid(e.A()*(mean_eig_scaling/mean_eig_size), e.center()))
    #sort by size
    idxs = np.argsort([s.Volume() for s in seed_ellipses])[::-1]
    hs = [seed_points[i] for i in idxs]
    se = [seed_ellipses_scaled[i] for i in idxs]
    return hs, se

N = 300
pts, _ = sample_cfree_handle(N, 3000,[])
adj_mat = vgraph_builder(pts, [], True)
from clique_covers import compute_greedy_edge_clique_cover, compute_greedy_clique_partition
cliques, M = compute_greedy_edge_clique_cover(adj_mat)
nr_cliques = len(cliques)

from pydrake.all import VPolytope, HPolyhedron
from scipy.spatial import ConvexHull

seed_ellipses = []
for c in cliques:
    pts_c =  pts[c, :]
    HE= get_lj_ellipse(pts_c)
    seed_ellipses.append(HE)
    
seed_polys = []
for c in cliques:
    if len(c)>= 3:
        pts_clique = pts[c,:]
        hull = ConvexHull(pts_clique)
        hull_vertices = pts_clique[hull.vertices, :]
        seed_polys.append(VPolytope(hull_vertices.T))
    else:
        seed_polys.append(None)
	
seed_points, metrics = get_kernel_iris_metrics([pts[c,:] for c in cliques])

from utils import generate_random_colors
from seeding_utils import sorted_vertices
from ellipse_utils import plot_ellipse
fig,ax = plt.subplots(figsize = (20,20))
ax.scatter(pts[:, 0], pts[:,1], c = 'k', s = 5)
#ax.scatter(pts[ind_set_idx, 0], pts[ind_set_idx,1], c = 'r', s = 10, zorder = 10)
# for k in kernels:
#     ax.scatter(pts[k, 0], pts[k, 1], s = 200)

colors = generate_random_colors(len(cliques))
for idx, c in enumerate(cliques):
    colc = colors[idx]
    seed_pol = seed_polys[idx]
    seed_ell = seed_ellipses[idx]
    scatter_plot = ax.scatter(pts[c, 0], pts[c, 1], s = 20, color = colc)
    #add edges from the center of each clique
    center = c[0]
    # for ci in c[1:]:
    #     ax.plot([pts[center,0], pts[ci,0]],[pts[center,1], pts[ci,1]], linewidth=1, c = colc)

    if seed_pol is not None:    
        v = sorted_vertices(seed_pol).T#s
        v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
        p = ax.plot(v[:,0], v[:,1], linewidth = 2, alpha = 0.2, c = colc, zorder = 0)

        ax.fill(v[:,0], v[:,1], alpha = 0.1, c = p[0].get_color(), zorder = 0)
    if seed_ell is not None:plot_ellipse(ax, seed_ell, 50, color = colc)
world.plot_cfree(ax)
plt.show()
print('done')