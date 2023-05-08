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