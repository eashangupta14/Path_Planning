
"""
Voronoi Road Map Planner
author: Atsushi Sakai (@Atsushi_twi)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Voronoi
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from matplotlib.patches import Rectangle

from Dijkstra_2 import DijkstraSearch

show_animation = False

class VoronoiRoadMapPlanner:

    def __init__(self, main_class):
        # parameter
        self.N_KNN = 30  # number of edge from one sampled point
        self.MAX_EDGE_LEN = 30.0  # [m] Maximum edge length
        self.coll_checker = main_class

    def planning(self, sx, sy, gx, gy, ox, oy, robot_radius):
        obstacle_tree = cKDTree(np.vstack((ox, oy)).T)

        sample_x, sample_y = self.voronoi_sampling(sx, sy, gx, gy, ox, oy)
        if show_animation:  # pragma: no cover
            plt.plot(sample_x, sample_y, ".b")

        road_map_info = self.generate_road_map_info(sample_x, sample_y, robot_radius, obstacle_tree)

        rx, ry = DijkstraSearch(show_animation,self.coll_checker).search(sx, sy, gx, gy,sample_x, sample_y,road_map_info)
        return rx, ry

    def is_collision(self, sx, sy, gx, gy, rr, obstacle_kd_tree):
        x = sx
        y = sy
        dx = gx - sx
        dy = gy - sy
        yaw = math.atan2(gy - sy, gx - sx)
        d = math.hypot(dx, dy)

        if d >= 50:
            return True,0

        D = rr
        n_step = round(d / D)

        for i in range(n_step):
            if self.coll_checker.is_collision(x, y, yaw):
                return True,0
            x += D * math.cos(yaw)
            y += D * math.sin(yaw)

        #  goal point check
        if self.coll_checker.is_collision(gx, gy, yaw):
            return True,0
        
        return False, yaw  # OK

    def generate_road_map_info(self, node_x, node_y, rr, obstacle_tree):
        """
        Road map generation

        node_x: [m] x positions of sampled points
        node_y: [m] y positions of sampled points
        rr: Robot Radius[m]
        obstacle_tree: KDTree object of obstacles
        """
        road_map = []
        n_sample = len(node_x)
        node_tree = cKDTree(np.vstack((node_x, node_y)).T)

        for (i, ix, iy) in zip(range(n_sample), node_x, node_y):

            dists, indexes = node_tree.query([ix, iy], k=n_sample)

            edge_id = []

            for ii in range(1, len(indexes)):
                nx = node_x[indexes[ii]]
                ny = node_y[indexes[ii]]

                collide,yaw = self.is_collision(ix, iy, nx, ny, rr, obstacle_tree)
                if not collide:
                    edge_id.append([indexes[ii],yaw*180/np.pi])

                if len(edge_id) >= self.N_KNN:
                    break

            road_map.append(edge_id)

        #  plot_road_map(road_map, sample_x, sample_y)

        return road_map

    @staticmethod
    def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

        for i, _ in enumerate(road_map):
            for ii in range(len(road_map[i])):
                ind = road_map[i][ii]

                plt.plot([sample_x[i], sample_x[ind]],
                         [sample_y[i], sample_y[ind]], "-k")

    @staticmethod
    def voronoi_sampling(sx, sy, gx, gy, ox, oy):
        oxy = np.vstack((ox, oy)).T

        # generate voronoi point
        vor = Voronoi(oxy)
        sample_x = [ix for [ix, _] in vor.vertices]
        sample_y = [iy for [_, iy] in vor.vertices]
        # plt.scatter(sample_x,sample_y,c="black")

        sample_x.append(sx)
        sample_y.append(sy)
        sample_x.append(gx)
        sample_y.append(gy)

        return sample_x, sample_y


def safe_main(start,end,grid_sz,obstacles,main_class):
    print(__file__ + " start!!")

    robot_size = 25*np.sqrt(2)  # Radius of Curvature

    sx = start[0]  # [m]
    sy = start[1]  # [m]
    gx = end[0]  # [m]
    gy = end[1]  # [m]
    grid_size = grid_sz  # [m]
    
    ox, oy = [], []

    ox = obstacles[0]
    oy = obstacles[1]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")
    
    rx, ry = VoronoiRoadMapPlanner(main_class).planning(sx, sy, gx, gy, ox, oy,robot_size)

    assert rx, 'Cannot found path'

    for i in range(len(rx)-1):
        x,y = rx[i],ry[i]
        xn,yn = rx[i+1],ry[i+1]
        dist = np.hypot(x-xn,y-yn)
        theta = np.arctan2(yn-y,xn-x)
        width = 50
        length = 50 + dist
        x_ = x - 25*np.sqrt(2)*np.cos(np.pi/4+theta)
        y_ = y - 25*np.sqrt(2)*np.sin(np.pi/4+theta)
        rect = Rectangle((x_, y_), length, width, angle=theta*180/np.pi,alpha = 0.5, linewidth=1, edgecolor='black', facecolor='yellow')
        ax.add_patch(rect)    
    plt.plot(rx, ry, "-r")
    #plt.pause(0.001)
    plt.show()
