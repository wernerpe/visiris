import numpy as np

from meshcat import Visualizer
from meshcat.geometry import Box, Sphere, Cylinder, MeshLambertMaterial, TriangularMeshGeometry
from meshcat.transformations import translation_matrix, rotation_matrix
from matplotlib.colors import to_hex as _to_hex
from pydrake.all import HPolyhedron
import mcubes
from  pydrake.all import ( 
                          Rgba,TriangleSurfaceMesh, SurfaceTriangle,
                          MathematicalProgram, SnoptSolver, le)
import colorsys
import itertools
from fractions import Fraction
import random
from functools import partial
from pydrake.all import VisibilityGraph,SceneGraphCollisionChecker

to_hex = lambda rgb: '0x' + _to_hex(rgb)[1:]

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

class EnvironmentVisualizer(Visualizer):

    def __init__(self):
        super().__init__()
        self['/Background'].set_property('visible', False)
        self.L = []
        self.U = []

    def cube(self, name, c, r, color=(1,0,0), opacity=1):
        c = np.array(c)
        self.L.append(c - r)
        self.U.append(c + r)
        return _cube(self, name, c, r, color, opacity)

    def box(self, name, l, u, color=(1,0,0), opacity=1):
        l = np.array(l)
        u = np.array(u)
        self.L.append(l)
        self.U.append(u)
        return _box(self, name, l, u, color, opacity)
    
    def box_visual(self, name, l, u, color=(1,0,0), opacity=1):
        l = np.array(l)
        u = np.array(u)
        #self.L.append(l)
        #self.U.append(u)
        return _box(self, name, l, u, color, opacity)

def _cube(vis, name, c, r, color, opacity):
    c = np.array(c, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    cube = vis[name]
    cube.set_object(Box(2 * r * np.ones(3)), material)
    cube.set_transform(translation_matrix(c))
    return cube

def _box(vis, name, l, u, color, opacity):
    l = np.array(l, dtype=float)
    u = np.array(u, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    box = vis[name]
    box.set_object(Box(u - l), material)
    c = (u - l) / 2
    box.set_transform(translation_matrix(l + c))
    return box

def _sphere(vis, name, pos, radius, color, opacity):
    pos = np.array(pos, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    box = vis[name]
    box.set_object(Sphere(radius), material)
    box.set_transform(translation_matrix(pos))
    return box

class Village(EnvironmentVisualizer):
    def __init__(self):
        super().__init__()
        self['/Grid'].set_property('visible', False)

    def ground(self, side, color=(1, 1, 1)):
        l = [-.5, -.5, -.02]
        u = [side + .5, side + .5, -.01]
        ground = self.box('ground', l, u, color)
        return ground

    def bush(self, name, c, r, h):
        c = np.array(c, dtype=float)
        l = [c[0] - r, c[1] - r, 0]
        u = [c[0] + r, c[1] + r, h]
        rand = 0.4*(np.random.rand(3)-0.5)
        color = (np.max([rand[0],0]), 0.5+rand[1], np.max([rand[2],0]))
        bush = self.box(name, l, u, color)
        return bush

    def tree(self, name, c, r, r_trunk):
        c = np.array(c, dtype=float)
        l = [c[0] - r_trunk, c[1] - r_trunk, 0]
        u = [c[0] + r_trunk, c[1] + r_trunk, c[2] - r]
        color = (.7, .35, 0)
        trunk = self.box(name + 'tree/_trunk', l, u, color)
        rand = 0.4*(np.random.rand(3)-0.5)
        color = (.2+rand[0], .8+rand[1], .2+rand[2])
        foliage = self.cube(name + 'tree/_foliage', c, r, color)
        return trunk, foliage

    def building(self, name, c, r, n):

        c = np.array(c, dtype=float)
        r = np.array(r, dtype=float)
        n = np.array(n, dtype=int)

        h = 2 * r[2]
        l = [c[0] - r[0], c[1] - r[1], 0]
        u = [c[0] + r[0], c[1] + r[1], h]
        color = (.8, .8, .8)
        body = self.box(name + 'building/_body', l, u, color)

        roof_ratio = 1 / 50
        l[2] = h 
        u[2] = h * (1 + roof_ratio)
        color = (.7, .0, .0)
        roof = self.box_visual(name + 'building/_roof', l, u, color)

        windows = []
        eps = 1e-2
        wcolor = (.3, .6, 1)
        fcolor = (0, 0, 0)
        d = 2 * r / n # Distance between windows in all directions.
        wr = d / 3.5 # Window radius in all directions.
        f = wr / 5 # Frame size in all directions.

        dw = np.array([wr[0], r[1] + 2 * eps, wr[2]])
        df = [f[0], - eps, f[2]]
        for i in range(n[0]):
            for j in range(n[2]):
                cij = np.array([c[0] - r[0] + d[0] * (i + .5), c[1], d[2] * (j + .5)])
                lij = cij - dw
                uij = cij + dw
                windows.append(self.box_visual(name + f'window/_window_x_{i}_{j}', lij, uij, wcolor))
                lij -= df
                uij += df
                windows.append(self.box_visual(name + f'window/_frame_x_{i}_{j}', lij, uij, fcolor))

        dw = np.array([r[0] + 2 * eps, wr[1], wr[2]])
        df = [- eps, f[1], f[2]]
        for i in range(n[1]):
            for j in range(n[2]):
                cij = np.array([c[0], c[1] - r[1] + d[1] * (i + .5), d[2] * (j + .5)])
                lij = cij - dw
                uij = cij + dw
                windows.append(self.box_visual(name + f'window/_window_y_{i}_{j}', lij, uij, wcolor))

                lij -= df
                uij += df
                windows.append(self.box_visual(name + f'window/_frame_y_{i}_{j}', lij, uij, fcolor))
        
        return body, roof, windows
    
    def build(self, village_height = 5, village_side = 19, building_every = 5, density = 0.3, seed = 12):
        np.random.seed(seed)
        self.density = density
        self.L_dom = [-.5, -.5, 0]
        self.U_dom = [village_side + .5, village_side + .5, village_height]

        assert (village_side + 1) % building_every == 0
        def direction():
            I = np.eye(2)
            directions = np.vstack((I, -I))
            return directions[np.random.randint(0, 4)]

        def walk(m):
            d1 = direction()
            starts = [np.zeros(2)]
            ends = [d1]
            blocks = [np.zeros(2), d1]
            for i in range(m):
                d2 = direction()
                blocks.append(blocks[-1] + d2)
                if all(d2 == d1):
                    ends[-1] += d1
                else:
                    starts.append(ends[-1])
                    ends.append(starts[-1] + d2)
                    d1 = d2
            return np.array(starts), np.array(ends), np.array(blocks)

        def _building(i, j, m):
            starts, ends, blocks = walk(m)
            offset = np.array([i, j]) + .5
            starts += offset
            ends += offset
            blocks += [i, j]
            for k, (s, e) in enumerate(zip(starts, ends)):
                c = (s + e) / 2
                d = np.abs(e - s) + 1
                r = list(.5 * d) + [village_height / 2]
                n = list(d) + [village_height]
                self.building(f'building/building_{i}_{j}_{k}', c, r, n)
            return blocks
                
        def _tree(i, j):
            name = f'tree/tree_{i}_{j}'
            r = .5 # Radius of foliage.
            r_trunk = .1
            low = [i + .5, j + .5, 1]
            high = [i + .5, j + .5, village_height - .5]
            c = np.random.uniform(low, high)
            self.tree(name, c, r, r_trunk)
                
        def _bush( i, j):
            name = f'bush/bush_{i}_{j}'
            r = np.random.uniform(low=.1, high=.35) # Radius.
            h = 4 * r # Height.
            c = np.array([i + .5, j + .5])
            self.bush(name, c, r, h)

        self.delete()
        
        # village ground
        ground_color = (.7, 1, .7)
        ground = self.ground(village_side, ground_color)

        # buildings
        blocks = []
        for i in range(village_side):
            for j in range(village_side):
                if (i + 1) % building_every == 0 and (j + 1) % building_every == 0:
                    blocks.append(_building(i, j, 4))
        blocks = [tuple(b) for b in np.vstack(blocks)]
            
        # trees and bushes
        for i in range(village_side):
            for j in range(village_side):
                if (i, j) not in blocks:
                    r = np.random.rand()
                    if r>1-self.density:
                        if r>1 - 0.5*self.density:
                            _bush(i, j)
                        else:
                            _tree(i,j)    
        self.obstacles = [HPolyhedron.MakeBox(l, u) for l, u in zip(self.L, self.U)]
        
        self.iris_domain = HPolyhedron.MakeBox(self.L_dom, self.U_dom)

        self.checker, self.vgraph_handle = self.to_drake_plant()



    def col_handle(self, pt):
        return not self.checker.CheckConfigCollisionFree(pt.squeeze())

    # def visible(self, a, b):
    #     if self.col_handle(a) or self.col_handle(b):
    #         return False
    #     if np.linalg.norm(a-b)<1e-5:
    #         return True
        
    #     tval = np.linspace(0, 1, 100)
    #     for t in tval:
    #         pt = (1-t)*a + t* b
    #         if self.col_handle(pt):
    #             return False
    #     return True
    
    def sample_cfree(self, n):
        points = []
        while len(points) < n:
            point = np.random.uniform(low=self.L_dom, high=self.U_dom)
            if self.point_in_cfree(point):
                points.append(point)
        return np.array(points)
    
    def point_in_cfree(self, pt):
        return self.checker.CheckConfigCollisionFree(pt.squeeze())
    
    def plot_points(self, points, color = (0,0,0), radius = 0.1):
        for i,pt in enumerate(points):
            _sphere(self, f"points/{i}", pt, radius, color, opacity=0.5)

            
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
            c = tuple([col/255 for col in colors[i]])#,opacity)
            prefix = f"/cfree/regions{region_suffix}/{i}"
            name = prefix + "/hpoly"
            if region.ambient_dimension() == 3:
                self.plot_hpoly3d(name, region,
                                  c, wireframe = wireframe, resolution = 30, opacity = opacity)
                
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
            return vertices, triangles#tri_drake

    def plot_hpoly3d(self, name, hpoly, color, wireframe = True, resolution = 30, opacity = 1.0):
            verts, triangles = self.get_plot_poly_mesh(hpoly,
                                                    resolution=resolution)
            
            
            box = self[name]
            col = to_hex(color)
            material = MeshLambertMaterial(color=col, opacity=opacity)
            box.set_object(TriangularMeshGeometry(verts, triangles), material)
    
    def convert_box(self, l, u):
        pos = 0.5*(l+u)
        size = u-l
        return pos, size
    
    def to_drake_plant(self, drone_size = 0.1):
        
        start = """<robot name="dronevillage">\n"""
        obstacles_string_start = """<link name="fixed">\n"""
        obstacles_strings =""""""
        i = 0
        for o_l, o_u in zip(self.L, self.U):
            cords, size = self.convert_box(o_l, o_u)
            obstacles_strings +="""    <collision name="obs_{}">
        <origin rpy="0 0 0" xyz="{} {} {}"/>
        <geometry><box size="{} {} {}"/></geometry>
    </collision>\n""".format(i, cords[0],cords[1], cords[2],size[0],size[1],size[2])
            i+=1

        obstacles_string_end = """</link>
<joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
</joint>\n"""
        robot_string =  """<link name="movable">
    <collision name="sphere">
    <geometry><sphere radius="{}"/></geometry>
    </collision>
</link>
<link name="for_joint"/>
<link name="for_joint2"/>
<joint name="x" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="{}" upper="{}"/>
    <parent link="world"/>
    <child link="for_joint"/>
</joint>
<joint name="y" type="prismatic">
    <axis xyz="0 1 0"/>
    <limit lower="{}" upper="{}"/>
    <parent link="for_joint"/>
    <child link="for_joint2"/>
</joint>
<joint name="z" type="prismatic">
    <axis xyz="0 0 1"/>
    <limit lower="{}" upper="{}"/>
    <parent link="for_joint2"/>
    <child link="movable"/>
</joint>\n""".format(drone_size, self.L_dom[0], self.U_dom[0], self.L_dom[1], self.U_dom[1], self.L_dom[2], self.U_dom[2])
            
    
  
        end = """</robot>"""
        urdf = start + obstacles_string_start  +obstacles_strings + obstacles_string_end + robot_string + end


        with open('tmp/village.urdf', 'w') as f:
            f.writelines(urdf)


        from pydrake.all import (RobotDiagramBuilder,
                                MeshcatVisualizerParams, 
                                Role,
                                MeshcatVisualizer,
                                StartMeshcat)
        builder = RobotDiagramBuilder()
        plant = builder.plant()
        scene_graph = builder.scene_graph()
        parser = builder.parser()
        models = [parser.AddModelFromFile('tmp/village.urdf')]
        plant.Finalize()
        meshcat2 = StartMeshcat()
        meshcat_params = MeshcatVisualizerParams()
        meshcat_params.role = Role.kProximity
        visualizer = MeshcatVisualizer.AddToBuilder(
                builder.builder(), scene_graph, meshcat2, meshcat_params)
        
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        diagram.ForcedPublish(diagram_context)
        plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
        robot_instances =[plant.GetModelInstanceByName("dronevillage")]
        checker = SceneGraphCollisionChecker(model = diagram.Clone(), 
                    robot_model_instances = robot_instances,
                    distance_function_weights =  [1] * plant.num_positions(),
                    #configuration_distance_function = _configuration_distance,
                    edge_step_size = 0.1)
        vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 

        return checker, vgraph_handle 



def vgraph(points, checker, parallelize):
    ad_mat = VisibilityGraph(checker.Clone(), np.array(points).T, parallelize = parallelize)
    N = ad_mat.shape[0]
    for i in range(N):
        ad_mat[i,i] = False
    #TODO: need to make dense for now to avoid wierd nx bugs for saving the metis file.
    return  ad_mat.toarray()