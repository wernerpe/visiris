# TODO: add my grid and star examples 

# dim = 2
# domain = HPolyhedron.MakeBox([-10.] * dim, [10.] * dim)
# def generate_grid_env(N = 10, seed = 10):
#     np.random.seed(10)
#     #N = 12
#     side_len = 10
#     spacing = side_len/(2*N + 2)
#     obstacles = []
#     for idx_x in range(N):
#         for idx_y in range(N):
#             center_x = 2*(2*spacing + 2*spacing*idx_x - side_len/2)
#             center_y = 2*(2*spacing + 2*spacing*idx_y - side_len/2)
#             p_min = [center_x - spacing/1., center_y - spacing/1.]
#             p_max = [center_x + spacing/1., center_y + spacing/1.]
#             obstacles.append(HPolyhedron.MakeBox(p_min, p_max))
    
#     iris_options = IrisOptions()
#     iris_options.require_sample_point_is_contained = True
#     iris_options.iteration_limit = 5
#     iris_options.termination_threshold = -1
#     iris_options.relative_termination_threshold = 0.05
    
#     def iris_mut(seed, obstacles, domain, iris_options):
#         return Iris(obstacles, seed, domain, iris_options)

#     iris_handle = partial(iris_mut, obstacles = obstacles,
#                           domain = domain, 
#                           iris_options = iris_options)
    
#     def morph_iris2(pt, regions, obstacles, domain, options):
#         obs = [o for o in obstacles]
#         for r in regions:
#             offset = 0.25*np.min(1/np.linalg.eig(r.MaximumVolumeInscribedEllipsoid().A())[0])
#             rnew = HPolyhedron(r.A(), r.b()-offset)
#             obs.append(rnew)
#         return Iris(obs, pt, domain, iris_options)

#     iris_handle2 = partial(morph_iris2, 
#                           obstacles = obstacles, 
#                           domain = domain, 
#                           options = iris_options) 

#     def point_in_regions(pt, regions): 
#         for r in regions:
#             if r.PointInSet(pt):
#                 return True
#         return False 

#     def collision(pt, obs, eps = 0.3):
#         for r in obs:
#             for idx in range(10):
#                 p = pt + eps*(np.random.rand(dim)-0.5)
#                 if r.PointInSet(p):
#                     return True
#         return False

#     col_handle = partial(collision, obs=obstacles)

#     def is_los(q1, q2, obs, domain):
#         if not domain.PointInSet(q1) or not domain.PointInSet(q2):
#             return False, []
#         tval = np.linspace(0,1, 40)
#         for t in tval:
#             pt = (1-t)*q1 + t* q2
#             if point_in_regions(pt, obs):
#                 return False, []
#         return True, []

#     los_handle = partial(is_los, obs = obstacles, domain = domain)
#     return None, obstacles, col_handle, los_handle, iris_handle, iris_handle2

# def point_in_regions(pt, regions): 
#         for r in regions:
#             if r.PointInSet(pt):
#                 return True
#         return False 
    
# def rand_colfree_point(col_hand):
#     good_sample = False #(not self.col_handle(pos_samp)) 
#     mindist = np.array([-10, -10])
#     minmaxdiff = np.array([20, 20])
#     for idx in range(10000):
#         rand = np.random.rand(2)
#         pos_samp = mindist + rand*minmaxdiff
        
#         if not col_handle(pos_samp):
#             return pos_samp
#     return None

# def get_cover_est(cover, col_handle, N = 10000):
#     ntot = 0
#     for idx in range(N):
#         pt=rand_colfree_point(col_handle)
#         if point_in_regions(pt, cover):
#             ntot +=1
#     return ntot*1.0/N


# def generate_star_env(points = 7, radius = 8, width = 0.9, pointrat = 0.3, seed = 15):
#     np.random.seed(seed)
#     angs = np.linspace(0, 2*np.pi, points+1)[:-1]
#     tip_vert = []
#     base_vert = []
#     for ang in angs:
#         tip = np.array([np.cos(ang), np.sin(ang)])
#         offsetdir = np.array([-np.sin(ang), np.cos(ang)])
#         tip_vert.append(tip*radius)
#         base_vert.append([tip*radius*pointrat + width*offsetdir, tip*radius*pointrat - width*offsetdir])

#     tips = []
#     for t, b in zip(tip_vert, base_vert):
#         tips.append(VPolytope(np.array([np.array(t)]+b)))
   
#     tip_poly = []
#     for tip in tips:
#         p = polytope.qhull(tip.vertices())
#         h = HPolyhedron(p.A, p.b)
#         tip_poly.append(h)
#     center_vs = []
#     for v in base_vert:
#         for vs in v:
#             center_vs.append(vs)
#     tip_verts = np.array(tip_vert)
#     p = polytope.qhull(np.array(center_vs))
#     cfree = [HPolyhedron(p.A, p.b)] + tip_poly
#     obstacles = []
#     for idx in range(len(tips)):
#         idx_n = (idx + 1)%len(tips)
#         verts = []
#         verts.append(tip_verts[idx])
#         verts.append(base_vert[idx][0])
#         verts.append(base_vert[idx_n][1])
#         verts.append(tip_verts[idx_n])
#         p = polytope.qhull(np.array(verts))
#         obstacles.append(HPolyhedron(p.A, p.b))

#     ##    
#     seeds_obs = np.array([0.6*(tip_vert[idx]+ tip_vert[(idx+1)%len(tip_vert)]) for idx in range(len(tip_vert))])
#     def iris_mut(seed, obstacles, domain, iris_options):
#         return Iris(obstacles, seed, domain, iris_options)

#     iris_handle = partial(iris_mut, obstacles = obstacles,
#                           domain = domain, 
#                           iris_options = iris_options)
#     def morph_iris2(pt, regions, obstacles, domain, options):
#         obs = [o for o in obstacles]
#         for r in regions:
#             offset = 0.25*np.min(1/np.linalg.eig(r.MaximumVolumeInscribedEllipsoid().A())[0])
#             rnew = HPolyhedron(r.A(), r.b()-offset)
#             obs.append(rnew)
#         return Iris(obs, pt, domain, iris_options)
    
#     iris_handle2 = partial(morph_iris2, 
#                           obstacles = obstacles, 
#                           domain = domain, 
#                           options = iris_options)
#     iris_handle_env = partial(morph_iris2, 
#                           obstacles = obstacles, 
#                           domain = domain, 
#                           options = iris_options) 
#     #obstacles = []
#     for s in seeds_obs:
#         obstacles.append(iris_handle2(s, obstacles))
        
#     def point_in_regions(pt, regions): 
#         for r in regions:
#             if r.PointInSet(pt):
#                 return True
#         return False 

#     def collision(pt, obs, eps = 0.3):
#         for r in obs:
#             for idx in range(10):
#                 p = pt + eps*(np.random.rand(dim)-0.5)
#                 if r.PointInSet(p):
#                     return True
#         return False

#     col_handle = partial(collision, obs=obstacles)

#     def is_los(q1, q2, obs, domain):
#         if not domain.PointInSet(q1) or not domain.PointInSet(q2):
#             return False, []
#         tval = np.linspace(0, 1, 40)
#         for t in tval:
#             pt = (1-t)*q1 + t* q2
#             if point_in_regions(pt, obs):
#                 return False, []
#         return True, []

#     los_handle = partial(is_los, obs = obstacles, domain = domain)
#     return cfree, obstacles, col_handle, los_handle, iris_handle, iris_handle2