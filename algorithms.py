import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from time import gmtime, strftime
from pydrake.geometry.optimization import HPolyhedron

def sample_node_pos(seeding_object,  MAXIT = 1e4):  
        it = 0
        good_sample = False
        while not good_sample and it < MAXIT:
            rand = np.random.rand(seeding_object.dim)
            pos_samp = seeding_object.min_pos + rand*seeding_object.min_max_diff 
            col = False
            for _ in range(10):
                r  = 0.01*(np.random.rand(seeding_object.dim)-0.5)
                col |= (seeding_object.col_handle(pos_samp+r) > 0)
            # if outside_regions: 
            #     good_sample = (not col) and (not seeding_object.point_in_guard_regions(pos_samp))
            # else:
            good_sample = (not col)
            it+=1
        if not good_sample:
            raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +" ERROR: Could not find collision free point in MAXIT %d".format(MAXIT))
        return pos_samp

def point_in_regions(regions,need_to_convert_samples, point_to_region_space, q):
    if need_to_convert_samples:
        pt = point_to_region_space(q)
    else:
        pt = q
    for r in regions:
        if r.PointInSet(pt):
            return True
    return False

def draw_connectivity_graph(seeding_object):
    fig = plt.figure()
    colors = []
    for idx in range(len(seeding_object.regions)):
        if idx<len(seeding_object.samples_to_connect):
            colors.append('c')
        elif idx<len(seeding_object.guard_regions):
            colors.append('b')
        else:
            colors.append('m')
    nx.draw_spring(seeding_object.connectivity_graph, 
                            with_labels = True, 
                            node_color = colors)

class SplittingVisSeeding:
    def __init__(self,
                 samples_to_connect,
                 limits = None,
                 alpha = 0.05,
                 eps = 0.05,
                 collision_handle = None,
                 is_in_line_of_sight = None,
                 iris_handle = None,
                 iris_handle_with_obstacles = None,
                 point_to_region_conversion = None,
                 plot_node = None,
                 plot_edge = None,
                 plot_region = None,
                 Verbose = True):

        self.verbose = Verbose
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.dim = len(self.min_pos)
        self.alpha = alpha
        self.eps = eps
        self.M = int(np.log(alpha)/np.log(1-eps))
        if self.verbose: 
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] Expecting points of interest in q')
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] GuardInsertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] {} probability that unseen region is less than {} "%" of Cfree '.format(1-self.alpha, 100*eps))


        self.col_handle = collision_handle
        self.is_in_line_of_sight = is_in_line_of_sight
        self.grow_region_at = iris_handle
        self.grow_region_at_with_obstacles = iris_handle_with_obstacles 
        self.guard_regions = []
        self.regions = []
        self.seed_points = []
        self.sample_set = {}
        self.samples_to_connect = samples_to_connect
        # self.sample_rank_handle = ranking_samples_handle
        self.nodes_to_connect = set([idx for idx, s in enumerate(self.samples_to_connect)])
        self.point_to_region_space = point_to_region_conversion
        self.need_to_convert_samples = True if self.point_to_region_space is not None else False



class VisSeeding:
    def __init__(self,
                 samples_to_connect,
                 limits = None,
                 alpha = 0.05,
                 eps = 0.05,
                 collision_handle = None,
                 is_in_line_of_sight = None,
                 iris_handle = None,
                 iris_handle_with_obstacles = None,
                 point_to_region_conversion = None,
                 plot_node = None,
                 plot_edge = None,
                 plot_region = None,
                 Verbose = True):

        self.verbose = Verbose
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.dim = len(self.min_pos)
        self.alpha = alpha
        self.eps = eps
        self.M = int(np.log(alpha)/np.log(1-eps))
        if self.verbose: 
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] Expecting points of interest in q')
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] GuardInsertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] {} probability that unseen region is less than {} "%" of Cfree '.format(1-self.alpha, 100*eps))


        self.col_handle = collision_handle
        self.is_in_line_of_sight = is_in_line_of_sight
        self.grow_region_at = iris_handle
        self.grow_region_at_with_obstacles = iris_handle_with_obstacles 
        self.guard_regions = []
        self.regions = []
        self.seed_points = []
        self.sample_set = {}
        self.samples_to_connect = samples_to_connect
        # self.sample_rank_handle = ranking_samples_handle
        self.nodes_to_connect = set([idx for idx, s in enumerate(self.samples_to_connect)])
        self.point_to_region_space = point_to_region_conversion
        self.need_to_convert_samples = True if self.point_to_region_space is not None else False

    def set_guard_regions(self, regions = None):
        if regions is None:
            if len(self.guard_regions)==0:
                self.regions = [self.grow_region_at(r) for r in self.samples_to_connect]
                self.seed_points = [s for s in self.samples_to_connect]
                for idx in range(len(self.regions)): self.guard_regions.append(idx) 
            else:
                raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[VISSeeding] guard_regions must be an empty list")
        else:
            for idx, r in enumerate(regions):
                seed, reg = r
                self.regions.append(reg)
                self.seed_points.append(seed)
                self.guard_regions.append(idx)

    def guard_phase(self,):
        it = 0
        while it < self.M:
            try:
                p = sample_node_pos(self, outside_regions=False)
            except:
                print(strftime("[%H:%M:%S] ", gmtime()) +"[VISSeeding] No sample found outside of regions ")
                break
            add_to_sample_set = False
            visible_regions = []
            p_region_space = self.point_to_region_space(p) if self.need_to_convert_samples else p
            for idx_guard in self.guard_regions: 
                guard_seed_point_q = self.seed_points[idx_guard]
                guard_seed_point_region_space = self.point_to_region_space(guard_seed_point_q) if self.need_to_convert_samples else guard_seed_point_q
                guard_region = self.regions[idx_guard]
                #check visibility in t
                if self.is_in_line_of_sight(p_region_space.reshape(-1,1), guard_seed_point_region_space.reshape(-1,1))[0]:
                    add_to_sample_set = True
                    visible_regions.append(guard_region)
            if add_to_sample_set:
                self.sample_set[str(p)] = [p, visible_regions]
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +"[VISSeeding] New guard placed N = ", str(len(self.guard_regions)), "it = ", it) 
                try:
                    #rnew = self.grow_region_at(p)
                    #self.regions.append(rnew)
                    self.seed_points.append(p)
                    self.guard_regions.append(len(self.seed_points)-1)
                    it = 0
                    #update visibility 
                    for s_key in self.sample_set.keys():
                        s_Q = self.sample_set[s_key][0]
                        if self.need_to_convert_samples:
                            s_conv = self.point_to_region_space(s_Q)
                        else:
                            s_conv = s_Q
                        if self.is_in_line_of_sight(s_conv.reshape(-1,1), p_region_space.reshape(-1,1))[0]:
                            self.sample_set[s_key][1].append(p)
                    if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] Sample set size',len(self.sample_set.keys()))
                except:
                    print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] Mosek failed, deleting point')
            it+=1
        #attempt to split the guards
        self.refine_gurads_greedy()
        
        #grow all guard regions
        for p in self.seed_points:
            self.grow_region_at(p)

        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.guard_regions)):
            self.connectivity_graph.add_node(idx)
            
        for idx1 in range(len(self.guard_regions)):
            for idx2 in range(idx1 +1, len(self.guard_regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)
        #if self.verbose: print("[VISSeeding] Connectivity phase")
    
    def get_new_seed_candidates(self, guard):
        vertices = self.compute_kernel_of_guard(guard)
        vertices += [self.seed_points[guard]]
        graph = nx.Graph()
        if len(vertices)>1:
            for i1, v1 in enumerate(vertices):
                for i2, v2 in enumerate(vertices):
                    if i1!=i2 and self.is_in_line_of_sight(v1.reshape(-1,1), v2.reshape(-1,1))[0]:
                        graph.add_edge(i1,i2)
            if len(graph.edges):
                new_cands = nx.maximal_independent_set(graph)
            else:
                raise ValueError("no edges detected, must be error")
            return [vertices[c] for c in new_cands]
        else:
            return [self.seed_points[guard]]

    def refine_gurads_greedy(self):
        continue_splitting = True
        while continue_splitting:
            candidate_splits = [self.get_new_seed_candidates(g) for g in self.guard_regions]
            best_split = max(candidate_splits, key = len)
            best_split_idx = candidate_splits.index(best_split) 
            if len(best_split)>1:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] Guard found to split into', len(best_split))
                #generate new regions
                nr = [self.grow_region_at(s) for s in best_split]
                self.regions += nr
                self.seed_points += best_split
                r_old = self.regions[best_split_idx]
                #keys_to_del = []
                gs_conv = [self.point_to_region_space(s) for s in best_split] if self.need_to_convert_samples else best_split
                
                for s_key in self.sample_set.keys():
                    vis_regs = self.sample_set[s_key][1] 
                    if r_old in vis_regs:
                        vis_regs.remove(r_old)
                    s_Q = self.sample_set[s_key][0]
                    if self.need_to_convert_samples:
                        s_conv = self.point_to_region_space(s_Q)
                    else:
                        s_conv = s_Q 
                    for idnr, gs in enumerate(gs_conv):
                        if self.is_in_line_of_sight(s_conv.reshape(-1,1), gs.reshape(-1,1))[0]:
                            vis_regs.append(nr[idnr])
                
                del self.seed_points[best_split_idx]
                del self.regions[best_split_idx]
                self.guard_regions = [self.regions.index(r) for r in self.regions]
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] No guard to split')
                continue_splitting = False
        #rebuild connectivity graph
        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.guard_regions)):
            self.connectivity_graph.add_node(idx)

        for idx1 in range(len(self.guard_regions)):
            for idx2 in range(idx1 +1, len(self.guard_regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)
        return

    def compute_kernel_of_guard(self, guard):
        ker = []
        for sampdat in self.sample_set.values():
            pos = sampdat[0]
            vis = [self.seed_points.index(sviz) for sviz in sampdat[1]]
            if len(vis)==1 and vis[0] == guard:
                ker.append(pos)
        return ker 

    def fill_remaining_space_phase(self,):
        it = 0
        while it < self.M:
            p = self.sample_node_pos(outside_regions=False)
            if point_in_regions(self.regions, self.need_to_convert_samples, self.point_to_region_space, p):
                it+=1
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +"[VISSeeding] New Region placed N = ", str(len(self.regions)), ", it = ", str(it)) 
                try:
                    rnew = self.grow_region_at_with_obstacles(p.reshape(-1, 1), self.regions)
                    self.regions.append(rnew)
                    self.seed_points.append(p)
                    it = 0
                    idx_new_region = len(self.regions)-1
                    self.connectivity_graph.add_node(idx_new_region)
                    #update connectivity graph
                    for idx, r in enumerate(self.regions[:-1]):
                        if r.IntersectsWith(rnew):
                            self.connectivity_graph.add_edge(idx, idx_new_region)
                except:
                    print(strftime("[%H:%M:%S] ", gmtime()) +'[VISSeeding] Mosek failed, deleting point')
        #get connected components
        components = [list(a) for a in nx.connected_components(self.connectivity_graph)]
        nodes_to_connect = self.nodes_to_connect if len(self.nodes_to_connect) else self.guard_regions
        for c in components:
            if set(nodes_to_connect) & set(c) == set(nodes_to_connect):
                return  True
        return False
    

class RandSeeding:
    def __init__(self,
                 samples_to_connect,
                 limits = None,
                 alpha = 0.05,
                 eps = 0.05,
                 collision_handle = None,
                 iris_handle = None,
                 iris_handle_with_obstacles = None,
                 point_to_region_conversion = None,
                 plot_node = None,
                 plot_edge = None,
                 plot_region = None,
                 terminate_early = True,
                 Verbose = True):

        self.verbose = Verbose
        self.limits = limits
        self.min_pos = self.limits[0]
        self.max_pos = self.limits[1]
        self.min_max_diff = self.max_pos - self.min_pos 
        self.dim = len(self.min_pos)
        self.alpha = alpha
        self.eps = eps
        self.M = int(np.log(alpha)/np.log(1-eps))
        if self.verbose: 
            print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] Expecting points of interest in q')
            print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] GuardInsertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] {} probability that unseen region is less than {} "%" of Cfree '.format(1-self.alpha, 100*eps))

        self.col_handle = collision_handle
        self.grow_region_at = iris_handle
        self.grow_region_at_with_obstacles = iris_handle_with_obstacles 
        self.regions = []
        self.seed_points = []
        self.samples_to_connect = samples_to_connect
        self.terminate_early = terminate_early
        if len(self.samples_to_connect) == 0:
            self.terminate_early = False
        self.nodes_to_connect = set([idx for idx, s in enumerate(self.samples_to_connect)])
        self.point_to_region_space = point_to_region_conversion
        self.need_to_convert_samples = True if self.point_to_region_space is not None else False

    def set_init_regions(self, regions = None):
        if regions is None:
            if len(self.regions)==0:
                self.regions = [self.grow_region_at(r) for r in self.samples_to_connect]
                self.seed_points = [s for s in self.samples_to_connect]
            else:
                raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[RandSeeding] regions must be an empty list")
        else:
            for idx, r in enumerate(regions):
                seed, reg = r
                self.regions.append(reg)
                self.seed_points.append(seed)
                
    def point_in_regions(self, q):
        if self.need_to_convert_samples:
            pt = self.point_to_region_space(q)
        else:
            pt = q
        for r in self.regions:
            if r.PointInSet(pt):
                    return True
        return False

    def sample_node_pos(self, MAXIT = 1e4):  
        rand = np.random.rand(self.dim)
        pos_samp = self.min_pos + rand*self.min_max_diff 
        good_sample = (not self.col_handle(pos_samp)) 
        it = 0
        while not good_sample and it < MAXIT:
            rand = np.random.rand(self.dim)
            pos_samp = self.min_pos + rand*self.min_max_diff 
            col = False
            for _ in range(10):
                r  = 0.01*(np.random.rand(self.dim)-0.5)
                col |= (self.col_handle(pos_samp+r) > 0)
            good_sample = not col
            it+=1
        if not good_sample:
            raise ValueError(strftime("[%H:%M:%S] ", gmtime()) +"[RandSeeding] ERROR: Could not find collision free point in MAXIT %d".format(MAXIT))
        return pos_samp

    def load_checkpoint(self, checkpoint):
        self.seed_points = checkpoint['seedpoints']
        A = checkpoint['regionsA']
        B = checkpoint['regionsB']
        self.regions = [HPolyhedron(a,b) for a,b, in zip(A,B)]
        #vis_reg = [[self.regions[idx] for idx in vis] for vis in checkpoint['sample_set_vis_regions']]
        #for i ,pt in enumerate(checkpoint['sample_set_points']):
        #    self.samples_outside_regions[str(pt)] = [pt, vis_reg[i]]
        
        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.regions)):
            self.connectivity_graph.add_node(idx)
            
        for idx1 in range(len(self.regions)):
            for idx2 in range(idx1 +1, len(self.regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)
        print(strftime("[%H:%M:%S] ", gmtime()) +"[RandSeeding] Checkpoint loaded successfully, current state is at end of guard phase")

    def run(self):
        self.set_init_regions()
        self.sample_regions_phase()        
        done_connecting = self.connectivity_phase()
            
    def sample_regions_phase(self,):
        it = 0
        self.connectivity_graph = nx.Graph()
        for idx in range(len(self.regions)):
            self.connectivity_graph.add_node(idx)

        for idx1 in range(len(self.regions)):
            for idx2 in range(idx1 +1, len(self.regions)):
                r1 = self.regions[idx1]
                r2 = self.regions[idx2]
                if r1.IntersectsWith(r2):
                    self.connectivity_graph.add_edge(idx1,idx2)

        while it < self.M:
            p = self.sample_node_pos()
            if self.point_in_regions(p):
                it+=1
            else:
                if self.verbose: print(strftime("[%H:%M:%S] ", gmtime()) +"[RandSeeding] New Region placed N = ", str(len(self.regions)), ", it = ", str(it)) 
                try:
                    rnew = self.grow_region_at_with_obstacles(p.reshape(-1, 1), self.regions)
                    self.regions.append(rnew)
                    self.seed_points.append(p)
                    it = 0
                    idx_new_region = len(self.regions)-1
                    self.connectivity_graph.add_node(idx_new_region)
                    #update connectivity graph
                    for idx, r in enumerate(self.regions[:-1]):
                        if r.IntersectsWith(rnew):
                            self.connectivity_graph.add_edge(idx, idx_new_region)
                    #check if all points in one connected component
                    #get connected components
                    components = [list(a) for a in nx.connected_components(self.connectivity_graph)]
                    #check if all nodes to connect are part of a single connected component
                    if self.terminate_early:
                        for c in components:
                            if set(self.nodes_to_connect) & set(c) == set(self.nodes_to_connect):
                                return True
                except:
                    print(strftime("[%H:%M:%S] ", gmtime()) +'[RandSeeding] Mosek failed, deleting point')
        
        components = [list(a) for a in nx.connected_components(self.connectivity_graph)]
        for c in components:
            if set(self.nodes_to_connect) & set(c) == set(self.nodes_to_connect):
                return True
        return False