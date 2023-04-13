#use this to load cg dataset
#https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2023/#problem-description
#need to figure out how to run iris in those polygons - maybe treat linesegments as polytopes and add as obstacles?

import numpy as np
import json
import shapely
import shapely.plotting
import triangle
from pydrake.all import HPolyhedron, VPolytope

def vert_list_to_numpy_array(vert_list):
	return np.array([[obj['x'], obj['y']] for obj in vert_list])

def shapely_polygon_to_triangle_package_dict(polygon):
	# Converts a shapely polygon (with no holes) into a dictionary that
	# can be used with the package triangle
	d = dict()
	n = len(polygon.exterior.coords) - 1
	d["vertices"] = polygon.exterior.coords[:-1]
	d["segments"] = np.vstack((np.arange(n), np.roll(np.arange(n), -1))).T
	return d

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
			self.name = data["name"]
			self.outer_boundary = shapely.Polygon(vert_list_to_numpy_array(data["outer_boundary"]))
			self.bbox = self.outer_boundary.bounds # minx, miny, maxx, maxy
			
			self.obstacle_segments = []
			self.obstacle_polygons = []
			self.obstacle_triangles = []
			for vert_list in data["holes"]:
				self._parse_obstacle(vert_list)
			
			self.cfree_polygon = shapely.Polygon(
				shell=self.outer_boundary.exterior.coords[:-1],
				holes=[poly.exterior.coords[:-1] for poly in self.obstacle_polygons]
			)

	def _parse_obstacle(self, vert_list):
		# vert_list is a list of dictionaries, with keys 'x' and 'y'.
		# Parse this, and append the appropriate object into 
		#  - self.obstacle_segments
		#  - self.obstacle_polygons
		#  - self.obstacle_triangles

		verts = vert_list_to_numpy_array(vert_list)
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

	def plot_cfree(self, ax):
		shapely.plotting.plot_polygon(self.cfree_polygon, ax=ax, add_points=False)

	def plot_boundary(self, ax):
		shapely.plotting.plot_polygon(self.outer_boundary, ax=ax, facecolor=(1,1,1,0), edgecolor="red", add_points=False)

	def plot_obstacles(self, ax):
		for poly in self.obstacle_polygons:
			shapely.plotting.plot_polygon(poly, ax=ax, facecolor="red", edgecolor="red", add_points=False)

	def sample_cfree(self, n):
		# Returns a uniform sample of n points in C-Free.
		pass # TODO

if __name__ == "__main__":
	world = World("./data/examples_01/cheese1028.instance.json")

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	world.plot_cfree(ax)
	plt.show()

	fig, ax = plt.subplots()
	world.plot_boundary(ax)
	world.plot_obstacles(ax)
	plt.show()