from tqdm import tqdm
from cgdataset import World, extract_small_examples
import numpy as np
import pickle
from scipy.sparse import lil_matrix
import os 
import time 

def create_visibility_graph(world, n, seed):
    np.random.seed(seed)
    points = np.zeros((n,2))
    adj_mat = lil_matrix((n,n))
    edge_endpoints = []
    for i in tqdm(range(n)):
        point = world.sample_cfree(1)[0]
		#ax.scatter([point[0]], [point[1]], color="black", s = 2)
        for j in range(len(points[:i])):
            other = points[j]
            if world.visible(point, other):
                edge_endpoints.append([[point[0], other[0]], [point[1], other[1]]])
                adj_mat[i,j] = adj_mat[j,i] = 1
        points[i] = point
    return points, adj_mat, edge_endpoints

def get_visibility_graph(world_name, world, n, seed):
    #check if precomputed
    assert world_name[0] != '.'
    assert world_name[-5:] == '.json'

    name = world_name[:-5]+f"_visgraph_{n}_{seed}.pkl"
    existing_visibility_graphs = os.listdir("./data/pre_generated_visibility_graphs/")
    if name in existing_visibility_graphs:
        print("example precomputed")
        with open("./data/pre_generated_visibility_graphs/"+name, 'rb') as f:
            dict = pickle.load(f)
        return dict['verts'], dict['adj'], dict['edge_endpoints'], dict['world_gen_time'], dict['vis_graph_gen_time']
    else:
        print("computing visgraph")
        path = "./data/examples_01/"+world_name
        name = world_name[:-5]+f"_visgraph_{n}_{seed}.pkl"
        t0 = time.time()
        t1 = time.time()
        points, adj_mat, edge_endpoints = create_visibility_graph(world, n, seed)
        t2 = time.time()
        with open("./data/pre_generated_visibility_graphs/"+name, 'wb') as f:
            pickle.dump({'verts':points, 'adj': adj_mat, 'edge_endpoints': edge_endpoints, 'world_gen_time': t1-t0, 'vis_graph_gen_time': t2-t1},f)
        return points, adj_mat, edge_endpoints, t1-t0, t2-t1

def create_visibility_graph_w_region_obstacles(world, n, seed, regions = None, shrunken_regions = None):
    np.random.seed(seed)
    points = np.zeros((n,2))
    adj_mat = lil_matrix((n,n))
    edge_endpoints = []
    if regions is not None:
        def point_in_regions(pt, regions): 
            for r in regions:
                if r.PointInSet(pt):
                    return True
            return False 
    
        def vis_reg(point, other):
            if not world.visible(point, other):
                return False
            else:
                tval = np.linspace(0, 1, 40)
                for t in tval:
                    pt = (1-t)*point + t* other
                    if point_in_regions(pt, shrunken_regions):
                        return False
            return True
        visibility_handle = vis_reg
    else:
        visibility_handle = world.visible

    for i in tqdm(range(n)):
        if regions is None:
            point = world.sample_cfree(1)[0]
        else:
            i_test =0
            found = False
            while i_test <100:
                point = world.sample_cfree(1)[0]
                if not point_in_regions(point, regions):
                    found = True
                    break
            if found == False:
                return points[:i], adj_mat[:i,:i], edge_endpoints
        #ax.scatter([point[0]], [point[1]], color="black", s = 2)
        for j in range(len(points[:i])):
            other = points[j]
            if visibility_handle(point, other):
                edge_endpoints.append([[point[0], other[0]], [point[1], other[1]]])
                adj_mat[i,j] = adj_mat[j,i] = 1
        points[i] = point
    return points, adj_mat, edge_endpoints

if __name__ == '__main__':
    
    #write all polys that are smaller than 2000 size into file with timing info
    extract_small_examples(1000)
    #n = 500
    seed = 0
    small_polys = []
    with open("./data/small_polys.txt") as f:
        for line in f:
            small_polys.append(line.strip())
    
    existing_visibility_graphs = os.listdir("./data/pre_generated_visibility_graphs/")

    for n in [450]:
        print("n = ", n)
        for idx, world_name in enumerate(small_polys): 
            print("example ", idx, "/", len(small_polys))
            name = world_name[:-5]+f"_visgraph_{n}_{seed}.pkl"
            if name not in existing_visibility_graphs:
                t0 = time.time()
                path = "./data/examples_01/"+world_name
                world = World(path, create_boundary_obstacles=False)
                t1 = time.time()
                points, adj_mat, edge_endpoints, _, tgraphgen = get_visibility_graph(world_name, world, n, seed)
                t2 = time.time()
                with open("./data/pre_generated_visibility_graphs/"+name, 'wb') as f:
                    pickle.dump({'verts':points, 'adj': adj_mat, 'edge_endpoints': edge_endpoints, 'world_gen_time': t1-t0, 'vis_graph_gen_time': tgraphgen},f)
