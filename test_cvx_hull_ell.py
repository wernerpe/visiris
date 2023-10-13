import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import MathematicalProgram, Solve, SolverOptions, CommonSolverOption
from cgdataset import World
from scipy.spatial import ConvexHull
from pydrake.all import VPolytope
from utils import generate_maximally_different_colors
from seeding_utils import sorted_vertices, point_near_regions, vis_reg
from scipy.sparse import lil_matrix
from tqdm import tqdm
import time
from clique_covers import compute_greedy_clique_partition, compute_greedy_clique_cover_w_ellipsoidal_convex_hull_constraint
from ellipse_utils import plot_ellipse_homogenous_rep

name_w = 'triangle'
world = World(f"data/evalexamples/{name_w}.json")
offs = -0.06
world.build_offset_cfree(offs)
def sample_cfree_handle(n, m, regions=None):
    points = np.zeros((n,2))
    if regions is None: regions = []		
    for i in range(n):
        bt_tries = 0
        while bt_tries<m:
            point = world.sample_cfree_distance(1, offs)[0]
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
            if  vis_reg(point, other, world, []):
                adj_mat[i,j] = adj_mat[j,i] = 1
    return adj_mat.toarray()

def seed_poly_hulls(cliques, pts):
    seed_polys = []
    for c in cliques:
        if len(c)>= 3:
            pts_clique = pts[c,:]
            hull = ConvexHull(pts_clique)
            hull_vertices = pts_clique[hull.vertices, :]
            seed_polys.append(VPolytope(hull_vertices.T))
        else:
            seed_polys.append(None)
    return seed_polys

def plot_polys(seed_polys, cliques, pts, ax, colors, zorder = 10):
    
    for idx, c in enumerate(cliques):
        colc = colors[idx]
        seed_pol = seed_polys[idx]
        scatter_plot = ax.scatter(pts[c, 0], pts[c, 1], s = 20, color = colc, zorder = zorder)
        #add edges from the center of each clique
        # for ci in c[1:]:
        #     ax.plot([pts[center,0], pts[ci,0]],[pts[center,1], pts[ci,1]], linewidth=1, c = colc)

        if seed_pol is not None:    
            v = sorted_vertices(seed_pol).T#s
            v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
            p = ax.plot(v[:,0], v[:,1], linewidth = 1, c = colc, zorder = zorder)

            ax.fill(v[:,0], v[:,1], alpha = 0.3, c = p[0].get_color(), zorder = zorder)
        else:
            ax.plot(pts[c, 0], pts[c, 1], linewidth = 5, c = colc, zorder = zorder)

seed = 25
N = 200
np.random.seed(seed)

import pickle
import os

name_vg= f"tmp/vg_test_cvx_hull_ell_{name_w}_{seed}_{N}.pkl"
if os.path.exists(name_vg):
    with open(name_vg, 'rb') as f:
        d = pickle.load(f)
        pts = d['pts']
        ad_mat = d['ad_mat']
else:
    pts, _ = sample_cfree_handle(N, 3000, [])
    ad_mat = vgraph_builder(pts)
    with open(name_vg, 'wb') as f:
        pickle.dump({'pts': pts, 'ad_mat': ad_mat}, f)

#dim_extra = 0
#pts = np.concatenate((pts, np.random.rand(len(pts), dim_extra).reshape(-1,dim_extra)), axis =1)

fig, axs= plt.subplots(1,4,  figsize = (24,8))
from clique_covers import compute_greedy_clique_partition_convex_hull
name_hyp = f"tmp/res_part_hyp_{N}_{seed}_{name_w}.pkl"
if os.path.exists(name_hyp):
    with open(name_hyp, 'rb') as f:
        d = pickle.load(f)
        cliques_hyp = d['cliques_hyp']
        t_hyp = d['t_hyp']
else:
    t1 = time.time()
    cliques_hyp = compute_greedy_clique_partition_convex_hull(ad_mat, pts, smin=1000, mode='full')
    t2 = time.time()
    t_hyp = t2-t1
    with open(name_hyp, 'wb') as f:
        pickle.dump({'cliques_hyp': cliques_hyp, 't_hyp': t_hyp }, f)

t1 = time.time()
cliques_hyp_new = compute_greedy_clique_partition_convex_hull(ad_mat, pts, smin=1000, alpha_max= 0.95*np.pi/2, mode='reduced') #
t2 = time.time()
t_hyp_new = t_hyp #t2-t1


t1 = time.time()
cliques_e, emats = compute_greedy_clique_cover_w_ellipsoidal_convex_hull_constraint(ad_mat, pts, smin=1000)
t2 = time.time()
t_e = t2-t1
t1 = time.time()
cliques = compute_greedy_clique_partition(ad_mat, smin = 1000)
t2 = time.time()
t_normal = t2-t1
names = [f"unconstrained {t_normal:.3f}s, {len(pts)} Vertices, {len(cliques)} Cliques, clique 1: {len(cliques[0])}", 
         f"ell constraint {t_e:.3f}s, {len(pts)} Vertices, {len(cliques_e)} Cliques, clique 1: {len(cliques_e[0])} ",
         f"hyp constraint {t_hyp:.3f}s, {len(pts)} Vertices, {len(cliques_hyp)} Cliques, clique 1: {len(cliques_hyp[0])}",
         f"hyp cons red {t_hyp_new:.3f}s, {len(pts)} Vertices, {len(cliques_hyp_new)} Cliques, clique 1: {len(cliques_hyp_new[0])}",
         ]
for ax, c, n in zip(axs, [cliques, cliques_e, cliques_hyp, cliques_hyp_new], names):
    colors = generate_maximally_different_colors(len(c)) #[(0,1,0)] if 'unconstrained' in n else [(0.07999999999999999, 0.8, 0.21090909090909077)] #
    world.plot_cfree_skel(ax)
    ax.scatter(pts[:, 0], pts[:, 1], c = 'k')
    ax.axis('equal')
    ax.set_title(n, fontsize = 10)
    seed_polys = seed_poly_hulls(c, pts[:,:2])
    plot_polys(seed_polys, c, pts, ax, colors, zorder= 10)
    # if 'ell' in n:
    #     for c, e in zip(colors, emats):
    #         plot_ellipse_homogenous_rep(ax,e,xrange=[-10, 10], yrange=[-6.5, 10], resolution=0.1, linewidth=2, color = c, zorder =9)

plt.show()

min_dists = []
for i in range(len(pts)):
    dists = np.linalg.norm(pts-pts[i,:], axis = 1)
print('done')