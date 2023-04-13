#use this to load cg dataset
#https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2023/#problem-description
#need to figure out how to run iris in those polygons - maybe treat linesegments as polytopes and add as obstacles?

import numpy as np
import json
import shapely
from pydrake.all import HPolyhedron

def vert_list_to_numpy_array(vert_list):
	return np.array([[obj['x'], obj['y']] for obj in vert_list])

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
			self.bbox = self.outer_boundary.bounds
			self.obstacle_segments = []
			self.obstacle_polygons = []
			self.obstacle_triangles = []
			for vert_list in data["holes"]:
				self._parse_obstacle(vert_list)
			self.cfree_polygon = self._compute_cfree_polygon()

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
		
		self.obstacle_polygons.append(shapely.Polygon(verts))

		# TODO: Populate self.obstacle_triangles

	def _compute_cfree_polygon(self):
		# Combines the outer boundary and holes into a single Shapely polygon
		pass # TODO

	def sample_cfree(self, n):
		# Returns a uniform sample of n points in C-Free.
		pass # TODO

if __name__ == "__main__":
	world = World("./data/examples_01/maze_001.instance.json")