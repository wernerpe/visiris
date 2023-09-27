import os
import pickle 
from visibility_graphs import get_visibility_graph
from pydrake.all import VPolytope, HPolyhedron

def create_experiment_directory(world_name, n, seed):
    dirname = world_name[:-5]
    dirname = "./experiment_logs/"+dirname
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Directory created successfully!")
    else:
        print("Directory already exists.")
    return

def experiment_string(world_name, n, seed, approach):
    return world_name[:-5]+f"_experiment_{n}_{seed}_{approach}"

def dump_experiment_results(world_name, experiment_name, chosen_verts, regions, tind, treg):
    with open("./experiment_logs/"+world_name[:-5] + "/" + experiment_name+".log", 'wb') as f:
        A = []
        b = []
        for r in regions:
            A.append(r.A())
            b.append(r.b())
        pickle.dump({'verts': chosen_verts, 'regions': [A,b], 'tind': tind, 'treg': treg}, f)

def dump_extended_experiment_results(world_name, experiment_name, chosen_verts, regions, coverage_ind_regions, coverage_with_fill, tind, treg, tfill, filltype):
    #filltype 0,1,2 connected components, most unutilized neighbours

    with open("./experiment_logs/"+world_name[:-5] + "/" + experiment_name+".log", 'wb') as f:
        A = []
        b = []
        for r in regions:
            A.append(r.A())
            b.append(r.b())
        pickle.dump({'verts': chosen_verts, 'regions': [A,b], 'tind': tind, 'treg': treg}, f)

def load_experiment(experiment_path, world_name, world, n, seed):
    with open("./experiment_logs/"+experiment_path, 'rb') as f:
        dict = pickle.load(f)
    points, adj_mat, edge_endpoints, twgen, tgraphgen = get_visibility_graph(world_name, world, n, seed)
    regions = [HPolyhedron(A,b) for A,b in zip(dict['regions'][0], dict['regions'][1])]

    timings = [tgraphgen, dict['tind'], dict['treg']]
    return points, adj_mat, edge_endpoints, dict['verts'], regions, timings


import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# def generate_random_colors(n):
#     # Get a list of all available named colors in matplotlib
#     named_colors = list(mcolors.CSS4_COLORS.keys())
#     total_named_colors = len(named_colors)

#     # Check if the requested number of colors is greater than the total available named colors
#     if n > total_named_colors:
#         print(f"Warning: Requested number of colors ({n}) is greater than the total available named colors ({total_named_colors}).")
#         print("Using random colors instead.")

#         # Generate random RGB colors
#         colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for _ in range(n)]
#     else:
#         # Shuffle the named colors list
#         random.seed(2)
#         random.shuffle(named_colors)

#         # Take the first N colors from the shuffled list
#         selected_colors = named_colors[:n]

#         # Convert the named colors to RGB values
#         colors = [mcolors.to_rgb(color) for color in selected_colors]

#     return colors

def generate_random_colors(n):
    # Get a list of all available named colors in matplotlib
    #named_colors = list(mcolors.CSS4_COLORS.keys())
    #total_named_colors = len(named_colors)

    # Check if the requested number of colors is greater than the total available named colors
    #if n > total_named_colors:
    #print(f"Warning: Requested number of colors ({n}) is greater than the total available named colors ({total_named_colors}).")
    #print("Using random colors instead.")

    # Generate random RGB colors
    colors = [(random.uniform(0, 0.8), random.uniform(0, 0.8), random.uniform(0, 1)) for _ in range(n)]
    # else:
    #     # Shuffle the named colors list
    #     random.seed(2)
    #     random.shuffle(named_colors)

    #     # Take the first N colors from the shuffled list
    #     selected_colors = named_colors[:n]

    #     # Convert the named colors to RGB values
    #     colors = [mcolors.to_rgb(color) for color in selected_colors]

    return colors


import colorsys

def generate_maximally_different_colors(n):
    """
    Generate n maximally different random colors for matplotlib.

    Parameters:
        n (int): Number of colors to generate.

    Returns:
        List of RGB tuples representing the random colors.
    """
    if n <= 0:
        raise ValueError("Number of colors (n) must be greater than zero.")

    # Define a list to store the generated colors
    colors = []

    # Generate n random hues, ensuring maximally different colors
    hues = [i / n for i in range(n)]
    random.seed(5)
    # Shuffle the hues to get random order of colors
    random.shuffle(hues)

    # Convert each hue to RGB
    for hue in hues:
        # We keep saturation and value fixed at 0.9 and 0.8 respectively
        saturation = 0.9
        value = 0.8
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    return colors

import numpy as np
from functools import partial
from pydrake.all import (
                         SurfaceTriangle, TriangleSurfaceMesh,
                         VPolytope, HPolyhedron,  Rgba)
import mcubes
from scipy.spatial import ConvexHull
import random
import colorsys


def plot_regions(meshcat, regions, ellipses = None,
                     region_suffix = '', colors = None,
                     wireframe = False,
                     opacity = 0.7,
                     fill = True,
                     line_width = 10,
                     darken_factor = .2,
                     el_opacity = 0.3,
                     resolution = 30,
                     offset = np.zeros(3)):
        if colors is None:
            colors = generate_maximally_different_colors(len(regions))

        for i, region in enumerate(regions):
            c = Rgba(*[col for col in colors[i]],opacity)
            prefix = f"/iris/regions{region_suffix}/{i}"
            name = prefix + "/hpoly"
            if region.ambient_dimension() == 3:
                # plot_hpoly3d(meshcat, name, region,
                #                   c, wireframe = wireframe, resolution = resolution, offset = offset)
                plot_hpoly3d_2(meshcat, name, region,
                                  c, wireframe = wireframe, resolution = resolution, offset = offset)

def get_plot_poly_mesh(region, resolution):

        def inpolycheck(q0, q1, q2, A, b):
            q = np.array([q0, q1, q2])
            res = np.min(1.0 * (A @ q - b <= 0))
            # print(res)
            return res

        aabb_max, aabb_min = get_AABB_limits(region)

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

def plot_hpoly3d(meshcat, name, hpoly, color, wireframe = True, resolution = 30, offset = np.zeros(3)):
        verts, triangles = get_plot_poly_mesh(hpoly,
                                                   resolution=resolution)
        meshcat.SetObject(name, TriangleSurfaceMesh(triangles, verts+offset.reshape(-1,3)),
                                color, wireframe=wireframe)
        
def plot_hpoly3d_2(meshcat, name, hpoly, color, wireframe = True, resolution = -1, offset = np.zeros(3)):
        #meshcat wierdness of double rendering
        hpoly = HPolyhedron(hpoly.A(), hpoly.b())
        verts = VPolytope(hpoly).vertices().T
        hull = ConvexHull(verts)
        triangles = []
        for s in hull.simplices:
            triangles.append(s)
        tri_drake = [SurfaceTriangle(*t) for t in triangles]
        # obj = self[name]
        # objwf = self[name+'wf']
        # col = to_hex(color)
        #material = MeshLambertMaterial(color=col, opacity=opacity)
        color2 = Rgba(0.8*color.r(), 0.8*color.g(), 0.8*color.b(), color.a())
        meshcat.SetObject(name, TriangleSurfaceMesh(tri_drake, verts+offset.reshape(-1,3)),
                                color, wireframe=False)
        meshcat.SetObject(name+'wf', TriangleSurfaceMesh(tri_drake, verts+offset.reshape(-1,3)),
                                color2, wireframe=True)
        meshcat.SetProperty(name[:-6], "visible", False)
        #meshcat.SetProperty(name, "visible", False)
        # #obj.set_object(TriangularMeshGeometry(verts, triangles), material)
        # material = MeshLambertMaterial(color=col, opacity=0.95, wireframe=True)
        # objwf.set_object(TriangularMeshGeometry(verts, triangles), material)