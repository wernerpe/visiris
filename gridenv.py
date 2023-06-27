import numpy as np
from pydrake.all import HPolyhedron, VPolytope
import shapely
import matplotlib.pyplot as plt
import shapely.plotting
from functools import partial
import mcubes
from  pydrake.all import (StartMeshcat, DiagramBuilder, AddMultibodyPlantSceneGraph, 
                          MeshcatVisualizer, Rgba,TriangleSurfaceMesh, SurfaceTriangle,
                          MathematicalProgram, SnoptSolver, le, RigidTransform, RollPitchYaw)
import colorsys
import itertools
from fractions import Fraction
import random


def infinite_hues():
    yield Fraction(0)
    for k in itertools.count():
        i = 2**k # zenos_dichotomy
        for j in range(1,i,2):
            yield Fraction(j,i)

def hue_to_hsvs(h: Fraction):
    # tweak values to adjust scheme
    for s in [Fraction(6,10)]:
        for v in [Fraction(6,10), Fraction(9,10)]:
            yield (h, s, v)

def rgb_to_css(rgb) -> str:
    uint8tuple = map(lambda y: int(y*255), rgb)
    return tuple(uint8tuple)

def css_to_html(css):
    return f"<text style=background-color:{css}>&nbsp;&nbsp;&nbsp;&nbsp;</text>"

def n_colors(n=33, rgbs_ret = False):
    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    csss = (rgb_to_css(rgb) for rgb in rgbs)
    to_ret = list(itertools.islice(csss, n)) if rgbs_ret else list(itertools.islice(csss, n))
    return to_ret

def n_colors_random(n=33, rgbs_ret = False):
    colors = n_colors(100 * n, rgbs_ret)
    return random.sample(colors, n)

def sorted_vertices(vpoly):
    assert vpoly.ambient_dimension() == 2
    poly_center = np.sum(vpoly.vertices(), axis=1) / vpoly.vertices().shape[1]
    vertex_vectors = vpoly.vertices() - np.expand_dims(poly_center, 1)
    sorted_index = np.arctan2(vertex_vectors[1], vertex_vectors[0]).argsort()[::-1]
    return vpoly.vertices()[:, sorted_index]

def postappend(verts):
    verts = np.concatenate((verts, verts[0,:].reshape(1,2)), axis=0)
    return [(v[0], v[1]) for v in verts]

class GridWorld:
    def __init__(self, N, side_len = 10, seed=1) -> None:
        np.random.seed(seed)
        dim = 2
        self.side_len = side_len
        spacing = 2*side_len/(2*N + 1)
        self.obstacles = []
        self.iris_domain = HPolyhedron.MakeBox([-side_len*1.0] * dim, [side_len*1.0] * dim)
        self.cfree_pieces = []
        for idx_x in range(N):
            for idx_y in range(N):
                #center_x = 2*(2*spacing + 2*spacing*idx_x - side_len/2)
                #center_y = 2*(2*spacing + 2*spacing*idx_y - side_len/2)
                p_min = [spacing*1.0 + idx_x*2*spacing -side_len, spacing*1.0 + idx_y*2*spacing - side_len]
                p_max = [spacing*2.0 + idx_x*2*spacing - side_len, spacing*2.0 + idx_y*2*spacing - side_len]
                self.obstacles.append(HPolyhedron.MakeBox(p_min, p_max))
        #shell_Verts =sorted_vertices(VPolytope(self.iris_domain)).T
        self.cfree_polygon = shapely.Polygon(shell =postappend(sorted_vertices(VPolytope(self.iris_domain)).T), 
                                             holes=[postappend(sorted_vertices(VPolytope(obs)).T) for obs in self.obstacles])
        #its not closed, but area is correct so fuck this - spent too long debuggin and cant be bothered
        # one_hole = shapely.Polygon(shell= postappend(sorted_vertices(VPolytope(self.obstacles[0])).T))
        # plt.figure()
        # ax = plt.gca()
        # shapely.plotting.plot_polygon(self.cfree_polygon, ax=ax, add_points=False)
        # shapely.plotting.plot_polygon(one_hole, ax=ax, add_points=False)
        # plt.show(block = False)
        #build domain regions
        for idx_x in range(N+1):
            x_min = idx_x*(2*spacing) - side_len
            x_max = spacing +idx_x*(2*spacing) - side_len
            self.cfree_pieces.append(HPolyhedron.MakeBox(np.array([x_min, -self.side_len]), np.array([x_max, self.side_len])))
            #build domain regions
        for idx_y in range(N+1):
            y_min = idx_y*(2*spacing) - side_len
            y_max = spacing +idx_y*(2*spacing) - side_len
            self.cfree_pieces.append(HPolyhedron.MakeBox(np.array([-self.side_len, y_min]), np.array([self.side_len, y_max])))
        print('done')

    def plot_cfree(self, ax):
        #shapely.plotting.plot_polygon(self.cfree_polygon, ax=ax, add_points=False)
        for obs in self.obstacles:
            v = sorted_vertices(VPolytope(obs)).T
            v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
            ax.fill(v[:,0], v[:,1], alpha = 1.0, c = 'k')

        for obs in self.cfree_pieces:
            v = sorted_vertices(VPolytope(obs)).T
            v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
            ax.fill(v[:,0], v[:,1], c = 'g', alpha =0.05)
            ax.plot(v[:,0], v[:,1], c = 'g', alpha =0.3)

    def col_handle(self, pt):
        for r in self.obstacles:
            if r.PointInSet(pt):
                return True
        return False
    
    def visible(self, a, b):
        if self.col_handle(a) or self.col_handle(b):
            return False
        if np.linalg.norm(a-b)<1e-5:
            return True
        
        tval = np.linspace(0, 1, 100)
        for t in tval:
            pt = (1-t)*a + t* b
            if self.col_handle(pt):
                return False
        return True
    
    def sample_cfree(self, n):
        points = []
        while len(points) < n:
            point = np.random.uniform(low=[-self.side_len, -self.side_len], high=[self.side_len, self.side_len])
            if self.point_in_cfree(point):
                points.append(point)
        return np.array(points)
    
    def point_in_cfree(self, pt):
        for r in self.cfree_pieces:
            if r.PointInSet(pt):
                return True
        return False
    
    def plot_HPoly(self, ax, HPoly, color = None):
        v = sorted_vertices(VPolytope(HPoly)).T#s
        v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
        if color is None:
            p = ax.plot(v[:,0], v[:,1], linewidth = 2, alpha = 0.7)
        else:
            p = ax.plot(v[:,0], v[:,1], linewidth = 2, alpha = 0.7, c = color)

        ax.fill(v[:,0], v[:,1], alpha = 0.5, c = p[0].get_color())


class GridWorld3D:
    def __init__(self, N, side_len = 10, seed=1) -> None:
        np.random.seed(seed)
        dim = 3
        self.side_len = side_len

        spacing = 2*side_len/(2*N + 1)
        self.spacing = spacing
        self.obstacles = []
        self.iris_domain = HPolyhedron.MakeBox([-side_len*1.0] * dim, [side_len*1.0] * dim)
        self.cfree_pieces = []
        for idx_x in range(N):
            for idx_y in range(N):
                 for idx_z in range(N):
                    #center_x = 2*(2*spacing + 2*spacing*idx_x - side_len/2)
                    #center_y = 2*(2*spacing + 2*spacing*idx_y - side_len/2)
                    p_min = [spacing*1.0 + idx_x*2*spacing -side_len, spacing*1.0 + idx_y*2*spacing - side_len, spacing*1.0 + idx_z*2*spacing - side_len]
                    p_max = [spacing*2.0 + idx_x*2*spacing - side_len, spacing*2.0 + idx_y*2*spacing - side_len,  spacing*2.0 + idx_z*2*spacing - side_len]
                    self.obstacles.append(HPolyhedron.MakeBox(p_min, p_max))
        #shell_Verts =sorted_vertices(VPolytope(self.iris_domain)).T
        # self.cfree_polygon = shapely.Polygon(shell =postappend(sorted_vertices(VPolytope(self.iris_domain)).T), 
        #                                      holes=[postappend(sorted_vertices(VPolytope(obs)).T) for obs in self.obstacles])
        #its not closed, but area is correct so fuck this - spent too long debuggin and cant be bothered
        # one_hole = shapely.Polygon(shell= postappend(sorted_vertices(VPolytope(self.obstacles[0])).T))
        # plt.figure()
        # ax = plt.gca()
        # shapely.plotting.plot_polygon(self.cfree_polygon, ax=ax, add_points=False)
        # shapely.plotting.plot_polygon(one_hole, ax=ax, add_points=False)
        # plt.show(block = False)
        #build domain regions
        for idx_x in range(N+1):
            x_min = idx_x*(2*spacing) - side_len
            x_max = spacing +idx_x*(2*spacing) - side_len
            self.cfree_pieces.append(HPolyhedron.MakeBox(np.array([x_min, -self.side_len, -self.side_len]), np.array([x_max, self.side_len, self.side_len])))
            #build domain regions
        for idx_y in range(N+1):
            y_min = idx_y*(2*spacing) - side_len
            y_max = spacing +idx_y*(2*spacing) - side_len
            self.cfree_pieces.append(HPolyhedron.MakeBox(np.array([-self.side_len, y_min, -self.side_len]), np.array([self.side_len, y_max, self.side_len])))
        for idx_z in range(N+1):
            z_min = idx_z*(2*spacing) - side_len
            z_max = spacing +idx_z*(2*spacing) - side_len
            self.cfree_pieces.append(HPolyhedron.MakeBox(np.array([-self.side_len, -self.side_len, z_min]), np.array([self.side_len, self.side_len, z_max])))
        
        self.meshcat = StartMeshcat()
        self.builder = DiagramBuilder()
        self.plant, scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.0)
        self.vis = MeshcatVisualizer.AddToBuilder(self.builder, scene_graph, self.meshcat)
        X_WC = RigidTransform(RollPitchYaw(0,0,0),np.array([2*side_len, 2*side_len, 0.85*side_len]) ) # some drake.RigidTransform()
        self.meshcat.SetTransform("/Cameras/default", X_WC) 
        self.plot_obstacles()
        self.plot_regions([self.iris_domain], colors = [(0,0,0)], opacity=0.1)
        print('done')


    def plot_obstacles(self):
        import meshcat
        from pydrake.all import Box, RigidTransform, RotationMatrix

        for i,o in enumerate(self.obstacles):
            c = o.ChebyshevCenter()
            box = Box(self.spacing, self.spacing,self.spacing)

            self.meshcat.SetObject(f"/drake/obstacles_{i}", box, Rgba(0.3,0.3,0.3,1))
            self.meshcat.SetTransform(f"/drake/obstacles_{i}",
                                  RigidTransform(RotationMatrix(),
                                                c))
            
    def plot_regions(self, regions, ellipses = None,
                     region_suffix = '', colors = None,
                     wireframe = False,
                     opacity = 0.7,
                     fill = True,
                     line_width = 10,
                     darken_factor = .2,
                     el_opacity = 0.3):
        if colors is None:
            colors = n_colors_random(len(regions), rgbs_ret=True)

        for i, region in enumerate(regions):
            c = Rgba(*[col/255 for col in colors[i]],opacity)
            prefix = f"/cfree/regions{region_suffix}/{i}"
            name = prefix + "/hpoly"
            if region.ambient_dimension() == 3:
                self.plot_hpoly3d(name, region,
                                  c, wireframe = wireframe, resolution = 30)
    def get_AABB_limits(self, hpoly, dim = 3):
        max_limits = []
        min_limits = []
        A = hpoly.A()
        b = hpoly.b()

        for idx in range(dim):
            aabbprog = MathematicalProgram()
            x = aabbprog.NewContinuousVariables(dim, 'x')
            cost = x[idx]
            aabbprog.AddCost(cost)
            aabbprog.AddConstraint(le(A@x,b))
            solver = SnoptSolver()
            result = solver.Solve(aabbprog)
            min_limits.append(result.get_optimal_cost()-0.01)
            aabbprog = MathematicalProgram()
            x = aabbprog.NewContinuousVariables(dim, 'x')
            cost = -x[idx]
            aabbprog.AddCost(cost)
            aabbprog.AddConstraint(le(A@x,b))
            solver = SnoptSolver()
            result = solver.Solve(aabbprog)
            max_limits.append(-result.get_optimal_cost() + 0.01)
        return max_limits, min_limits

    def get_plot_poly_mesh(self, region, resolution):

            def inpolycheck(q0, q1, q2, A, b):
                q = np.array([q0, q1, q2])
                res = np.min(1.0 * (A @ q - b <= 0))
                # print(res)
                return res

            aabb_max, aabb_min = self.get_AABB_limits(region)

            col_hand = partial(inpolycheck, A=region.A(), b=region.b())
            vertices, triangles = mcubes.marching_cubes_func(tuple(aabb_min),
                                                            tuple(aabb_max),
                                                            resolution,
                                                            resolution,
                                                            resolution,
                                                            col_hand,
                                                            0.5)
            tri_drake = [SurfaceTriangle(*t) for t in triangles]
            return vertices, tri_drake

    def plot_hpoly3d(self, name, hpoly, color, wireframe = True, resolution = 30):
            verts, triangles = self.get_plot_poly_mesh(hpoly,
                                                    resolution=resolution)
            self.meshcat.SetObject(name, TriangleSurfaceMesh(triangles, verts),
                                    color, wireframe=wireframe)
        
    def plot_cfree(self, meshcat):
        #shapely.plotting.plot_polygon(self.cfree_polygon, ax=ax, add_points=False)
        for obs in self.obstacles:
            v = sorted_vertices(VPolytope(obs)).T
            v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
            ax.fill(v[:,0], v[:,1], alpha = 1.0, c = 'k')

        for obs in self.cfree_pieces:
            v = sorted_vertices(VPolytope(obs)).T
            v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
            ax.fill(v[:,0], v[:,1], c = 'g', alpha =0.05)
            ax.plot(v[:,0], v[:,1], c = 'g', alpha =0.3)

    def col_handle(self, pt):
        for r in self.obstacles:
            if r.PointInSet(pt):
                return True
        return False
    
    def visible(self, a, b):
        if self.col_handle(a) or self.col_handle(b):
            return False
        if np.linalg.norm(a-b)<1e-5:
            return True
        
        tval = np.linspace(0, 1, 100)
        for t in tval:
            pt = (1-t)*a + t* b
            if self.col_handle(pt):
                return False
        return True
    
    def sample_cfree(self, n):
        points = []
        while len(points) < n:
            point = np.random.uniform(low=[-self.side_len, -self.side_len, -self.side_len], high=[self.side_len, self.side_len, self.side_len])
            if self.point_in_cfree(point):
                points.append(point)
        return np.array(points)
    
    def point_in_cfree(self, pt):
        for r in self.cfree_pieces:
            if r.PointInSet(pt):
                return True
        return False
    
    def plot_HPoly(self, ax, HPoly, color = None):
        v = sorted_vertices(VPolytope(HPoly)).T#s
        v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
        if color is None:
            p = ax.plot(v[:,0], v[:,1], linewidth = 2, alpha = 0.7)
        else:
            p = ax.plot(v[:,0], v[:,1], linewidth = 2, alpha = 0.7, c = color)

        ax.fill(v[:,0], v[:,1], alpha = 0.5, c = p[0].get_color())


if __name__ == "__main__":
    seed = 1
    size = 5
    world = GridWorld(12, side_len=size, seed = seed)
    fig,ax = plt.subplots(figsize = (10,10))
    ax.set_xlim((-size, size))
    ax.set_ylim((-size, size))
    world.plot_cfree(ax)
    plt.pause(0.01)