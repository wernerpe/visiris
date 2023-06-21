import numpy as np
from pydrake.all import HPolyhedron, VPolytope
import shapely
import matplotlib.pyplot as plt
import shapely.plotting

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
                
if __name__ == "__main__":
    seed = 1
    size = 5
    world = GridWorld(12, side_len=size, seed = seed)
    fig,ax = plt.subplots(figsize = (10,10))
    ax.set_xlim((-size, size))
    ax.set_ylim((-size, size))
    world.plot_cfree(ax)
    plt.pause(0.01)