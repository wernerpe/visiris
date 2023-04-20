#use this to load cg dataset
#https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2023/#problem-description
#need to figure out how to run iris in those polygons - maybe treat linesegments as polytopes and add as obstacles?

import numpy as np
import json
import shapely
import shapely.plotting
import triangle
from pydrake.all import HPolyhedron, VPolytope

def vert_list_to_numpy_array(vert_list, scaler = 1.0):
	return np.array([[scaler*obj['x'], scaler*obj['y']] for obj in vert_list])

def shapely_polygon_to_triangle_package_dict(polygon):
	# Converts a shapely polygon (with no holes) into a dictionary that
	# can be used with the package triangle
	d = dict()
	n = len(polygon.exterior.coords) - 1
	d["vertices"] = polygon.exterior.coords[:-1]
	d["segments"] = np.vstack((np.arange(n), np.roll(np.arange(n), -1))).T
	return d

def sorted_vertices(vpoly):
    assert vpoly.ambient_dimension() == 2
    poly_center = np.sum(vpoly.vertices(), axis=1) / vpoly.vertices().shape[1]
    vertex_vectors = vpoly.vertices() - np.expand_dims(poly_center, 1)
    sorted_index = np.arctan2(vertex_vectors[1], vertex_vectors[0]).argsort()
    return vpoly.vertices()[:, sorted_index]

class World():
	# Encapsulates all information about a world:
	#  - Name
	#  - Outer Boundary
	#  - Bounding Box
	#  - Obstacles as Line Segments
	#  - Obstacles as Shapely Polygons
	#  - Obstacles as Triangles (HRep Polyhedra)
	#  - C-Free as a Shapely Polygon
	def __init__(self, fname):
		# Load the world specified by the given dataset
		with open(fname) as file:
			data = json.load(file)
			self.data = data
			self.name = data["name"]
			self.scaler = 1.0
			maxdat = [np.max([ob['x'], ob['y']]) for ob in data["outer_boundary"]]
			max = 0.1*np.max(maxdat)
			self.scaler = 1.0/max 
			self.outer_boundary = shapely.Polygon(vert_list_to_numpy_array(data["outer_boundary"], self.scaler))
			self.bounds = self.outer_boundary.bounds # minx, miny, maxx, maxy
			
			self.obstacle_segments = []
			self.obstacle_polygons = []
			self.obstacle_triangles = []
			for vert_list in data["holes"]:
				self._parse_obstacle(vert_list)
			
			self.cfree_polygon = shapely.Polygon(
				shell=self.outer_boundary.exterior.coords[:-1],
				holes=[poly.exterior.coords[:-1] for poly in self.obstacle_polygons]
			)
			self.iris_domain = None
			self.domain_obstacles = self._create_boundary_obstacles()
			
	def _parse_obstacle(self, vert_list):
		# vert_list is a list of dictionaries, with keys 'x' and 'y'.
		# Parse this, and append the appropriate object into 
		#  - self.obstacle_segments
		#  - self.obstacle_polygons
		#  - self.obstacle_triangles

		verts = vert_list_to_numpy_array(vert_list, self.scaler)
		for i in range(0, len(verts)):
			j = (i+1) % len(verts)
			self.obstacle_segments.append(verts[[i,j]])
		
		poly = shapely.Polygon(verts)
		self.obstacle_polygons.append(poly)

		tris = triangle.triangulate(shapely_polygon_to_triangle_package_dict(poly), "p")
		# import matplotlib.pyplot as plt
		# triangle.plot(plt.axes(), **tris)
		# plt.show()
		delaunay_verts = np.array(tris["vertices"].tolist())
		for tri_idx in tris["triangles"].tolist():
			tri_points = delaunay_verts[tri_idx].T # Drake wants the points to be columns
			self.obstacle_triangles.append(HPolyhedron(VPolytope(tri_points)))
	
	def _create_boundary_obstacles(self,):
		#create AABB surrounding outer boundary of Polygon
		vertsAABB = [[1.1*self.bounds[0]-1,1.1*self.bounds[1]-1],
	   			 [1.1*self.bounds[2],1.1*self.bounds[1]-1],
	   			 [1.1*self.bounds[2],1.1*self.bounds[3]],
	   			 [1.1*self.bounds[0]-1,1.1*self.bounds[3]],
	   			]
		boundary_verts = vert_list_to_numpy_array(self.data["outer_boundary"], self.scaler)
		
		verts = np.vstack((np.array(vertsAABB), boundary_verts))
		self.iris_domain = HPolyhedron(VPolytope(verts.T))

		i = np.arange(4)
		segAABB = np.stack([i, i + 1], axis=1) % 4
		i = np.arange(boundary_verts.shape[0])
		seg_boundary = np.stack([i, i + 1], axis=1) % boundary_verts.shape[0]
		seg = np.vstack([segAABB, seg_boundary + segAABB.shape[0]])
		interior_pt = self.sample_cfree(1)[0]
		
		#create triangular obstacles
		TR = dict(vertices=verts, segments=seg, holes=[list(interior_pt)])
		tris = triangle.triangulate(TR, 'p')
		delaunay_verts = np.array(tris["vertices"].tolist())
		for tri_idx in tris["triangles"].tolist():
			tri_points = delaunay_verts[tri_idx].T # Drake wants the points to be columns
			self.obstacle_triangles.append(HPolyhedron(VPolytope(tri_points)))

	def plot_cfree(self, ax):
		shapely.plotting.plot_polygon(self.cfree_polygon, ax=ax, add_points=False)

	def plot_boundary(self, ax):
		shapely.plotting.plot_polygon(self.outer_boundary, ax=ax, facecolor=(1,1,1,0), edgecolor="red", add_points=False)

	def plot_triangles(self, ax):
		for tri in self.obstacle_triangles:
			v = VPolytope(tri).vertices().T
			v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
			ax.plot(v[:,0], v[:,1], alpha = 0.1, c = 'k')

	def plot_HPoly(self, ax, HPoly):
		v = sorted_vertices(VPolytope(HPoly)).T#s
		v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
		p = ax.plot(v[:,0], v[:,1], linewidth = 2, alpha = 0.7)
		ax.fill(v[:,0], v[:,1], alpha = 0.5, c = p[0].get_color())

	def plot_obstacles(self, ax):
		for poly in self.obstacle_polygons:
			shapely.plotting.plot_polygon(poly, ax=ax, facecolor="red", edgecolor="red", add_points=False)

	def sample_cfree(self, n):
		points = []
		while len(points) < n:
			point = np.random.uniform(low=self.bounds[0:2], high=self.bounds[2:4])
			if self.cfree_polygon.contains(shapely.Point(point)):
				points.append(point)
		return np.array(points)

	def visible(self, p, q):
		# Returns True if p and q can see each other
		# If either p or q are in an obstacle, returns False
		if not self.cfree_polygon.contains(shapely.Point(p)):
			return False
		if not self.cfree_polygon.contains(shapely.Point(q)):
			return False
		l = shapely.LineString([p, q])
		return not shapely.crosses(self.cfree_polygon, l)

	def cfree_area(self):
		return self.cfree_polygon.area

if __name__ == "__main__":
	world = World("./data/examples_01/srpg_iso_aligned_mc0000172.instance.json")

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	world.plot_cfree(ax)
	plt.draw()
	plt.pause(0.001)

	points = []
	for i in range(100):
		point = world.sample_cfree(1)[0]
		ax.scatter([point[0]], [point[1]], color="black")
		for other in points:
			if world.visible(point, other):
				ax.plot([point[0], other[0]], [point[1], other[1]], color="black", linewidth=0.25)
		points.append(point)
		plt.draw()
		plt.pause(0.001)

	plt.show()