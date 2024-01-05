# Utility class 
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate
from A_Star_2 import greedy_main
from Voronoi_2 import safe_main
from PRM import prm_main
from RRT import rrt_main
class pathplanner(object):
	
	def __init__(self):
		self.make_arena()
		self.robot = Polygon([(0, 0), (0, 50), (50, 50), (50, 0)])

	def make_arena(self):
		wall1 = Polygon([(199, 300), (199, 100), (201, 100), (201, 300)])
		wall2 = Polygon([(100, 101), (100, 99), (300, 99), (300, 101)])
		wall3 = Polygon([(500, 101), (500, 99), (700, 99), (700, 101)])
		wall4 = Polygon([(599, 300), (599, 100), (601, 100), (601, 300)])
		wall5 = Polygon([(399, 200), (401, 200), (401, 0), (399, 0)])

		outwall1 = Polygon([(0,0),(800,0),(800,-1),(0,-1)])
		outwall2 = Polygon([(0,300),(800,300),(800,301),(0,301)])
		outwall3 = Polygon([(0,0),(0,300),(-1,300),(-1,0)])
		outwall4 = Polygon([(800,0),(800,300),(801,300),(801,0)])

		self.obstacles = [wall1, wall2,wall3,wall4,wall5,outwall1,outwall2,outwall3,outwall4]

		ox = []
		oy = []
		for i in range(800):
			ox.append(i)
			oy.append(0)
		for i in range(800):
			ox.append(i)
			oy.append(300)
		for j in range(300):
			ox.append(0)
			oy.append(j)
		for j in range(300):
			ox.append(800)
			oy.append(j)

		for j in range(100,300):
			ox.append(200)
			oy.append(j)
		for j in range(200):
			ox.append(400)
			oy.append(j)
		for j in range(100,300):
			ox.append(600)
			oy.append(j)
		for i in range(100,300):
			ox.append(i)
			oy.append(100)
		for i in range(500,700):
			ox.append(i)
			oy.append(100)

		self.obs = [ox,oy]

	def is_collision(self,x,y,theta):
		#print(self.robot)
		bot_translate = translate(self.robot, x-25, y-25)
		bot_rotate = rotate(bot_translate, theta, origin='centroid')
		return any(bot_rotate.intersects(obstacle) for obstacle in self.obstacles)

	def plot_cs(self, theta, grid_sz = 2):
		free_point = []
		col_point = []
		for x in range(0, 800, grid_sz):
			for y in range(0, 300, grid_sz):
				collision = self.is_collision(x, y, theta)
				col_point.append([x,y]) if collision else free_point.append([x,y])

		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111)

		free_point = np.array(free_point)
		col_point = np.array(col_point)
		# Plot the cs
		plt.scatter(free_point[:,0],free_point[:,1], color = "green")
		plt.scatter(col_point[:,0],col_point[:,1], color = "red")

		# Plot the arena
		for obstacle in self.obstacles:
		    x, y = zip(*obstacle.exterior.coords)
		    ax.plot(x, y, color='black')

		ax.set_xlabel('X position')
		ax.set_ylabel('Y position')
		plt.axis('equal')
		plt.show()

	def greedy(self,start,destination,grid_sz = 2):
		# Algorithm from pythonrobotics. changes mentioned in report
		
		greedy_main(start,destination,grid_sz,self.obs,self)

	def safe(self, start, destination, grid_sz = 2):
		# Algorithm from pythonrobotics. changes mentioned in report
		safe_main(start,destination,grid_sz,self.obs,self)

	def PRM(self, start, destination, grid_sz = 2, num_samples = 50):
		prm_main(start, destination, grid_sz,self.obs,self,num_samples)

	def RRT(self, start, destination, grid_sz = 2, num_samples = 50):
		rrt_main(start, destination, grid_sz,self.obs,self,num_samples)


		
		
		

