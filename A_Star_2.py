
import math
import matplotlib.pyplot as plt
show_animation = False
from matplotlib.patches import Rectangle
import numpy as np

class AStarPlanner:

    def __init__(self, ox, oy, resolution, main_class):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.resolution = resolution
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.coll_checker = main_class
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)
        self.allow_theta = {0:{0:[[0]],45:[[1],[2,3,4]],90:[[1,2],[3,4]],135:[[1,2,3],[4]]},
                            45:{45:[[0]],0:[[1],[2,3,4]],90:[[2],[1,3,4]],135:[[2,3],[1,4]]},
                            90:{90:[[0]],0:[[2,1],[3,4]],45:[[2],[1,3,4]],135:[[3],[1,2,4]]},
                            135:{135:[[0]],0:[[1,2,3],[4]],45:[[2,3],[1,4]],90:[[3],[1,2,4]]}} 


    class Node:
        def __init__(self, x, y, cost, parent_index, orientation = 1000):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index
            self.orientation  = orientation

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),self.calc_xy_index(sy, self.min_y), 0.0, -1, [0,180])
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),self.calc_xy_index(gy, self.min_y), 0.0, -1, 1000)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        print('Start Planning')
        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set,key=lambda o: open_set[o].cost*0 + self.calc_heuristic(goal_node,open_set[o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                print("Final Cost:",current.cost)
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],current.y + self.motion[i][1],current.cost + self.motion[i][2], c_id, self.motion[i][3])
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                # If the motion is not allowed, do nothing
                if not self.verify_robot_constraint(current,node):
                    continue    

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def verify_robot_constraint(self,current,node):
        c_orient = current.orientation
        n_orient = node.orientation
        
        px = current.x
        py = current.y

        ini = c_orient[0]
        tar = n_orient[0]

        for a in self.allow_theta[ini][tar]:    
            for q in a:
                no = 0
                if q in self.obstacle_map[px][py][1]:
                    no = 1
                    break
            if no == 0:
                return True
        return False

        

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y][0]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        pty = 0
        self.obstacle_map = [[[False,[]] for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                print(pty)
                pty = pty + 1
                y = self.calc_grid_position(iy, self.min_y)
                if (self.coll_checker.is_collision(x,y,0) and 
                    self.coll_checker.is_collision(x,y,45) and 
                    self.coll_checker.is_collision(x,y,90) and 
                    self.coll_checker.is_collision(x,y,135)):
                    self.obstacle_map[ix][iy][0] = True
                else:
                    self.obstacle_map[ix][iy][1] = self.allowq(x,y)
                    #print(self.obstacle_map[ix][iy][1])
                    # break

    def allowq(self,x,y):

        theta = 0
        q = []
        while (theta<180):
            #print(theta)
            
            if self.coll_checker.is_collision(x,y,theta):
                #print(theta)
                #print((theta // 45))
                if theta not in [0,45,135,90]:
                    q.append((theta//45)+1)
                    theta = ((theta//45)+1)*45
                else:
                    if theta == 0:
                        q.append(4)
                        q.append(1)
                    else:
                        q.append((theta//45))
                        q.append((theta//45)+1)
                    theta = ((theta//45)+1)*45
            else:
                theta = theta + 5
            
        return q

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1, [0,180]],
                  [0, 1, 1, [90, 270]],
                  [-1, 0, 1, [0, 180]],
                  [0, -1, 1, [90, 270]],
                  [-1, -1, math.sqrt(2),[45,225]],
                  [-1, 1, math.sqrt(2),[135,315]],
                  [1, -1, math.sqrt(2),[135, 315]],
                  [1, 1, math.sqrt(2),[45,225]]]

        return motion


def greedy_main(start,end,grid_sz,obstacles, main_class):
    print(__file__ + " start!!")

    # start and goal position
    
    sx = start[0]  # [m]
    sy = start[1]  # [m]
    gx = end[0]  # [m]
    gy = end[1]  # [m]
    grid_size = grid_sz  # [m]
    # robot_radius = 1.0  # [m]

    # set obstacle positions
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

    a_star = AStarPlanner(ox, oy, grid_size, main_class)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    for i in range(len(rx)-1):
        x,y = rx[i],ry[i]
        xn,yn = rx[i+1],ry[i+1]
        dist = np.hypot(x-xn,y-yn)
        theta = np.arctan2(yn-y,xn-x)
        width = 50
        length = 50
        x_ = x - 25*np.sqrt(2)*np.cos(np.pi/4+theta)
        y_ = y - 25*np.sqrt(2)*np.sin(np.pi/4+theta)
        rect = Rectangle((x_, y_), length, width, angle=theta*180/np.pi,alpha = 0.5, linewidth=1, edgecolor='black', facecolor='yellow')
        ax.add_patch(rect)    
    plt.plot(rx, ry, "-r")
    #plt.pause(0.001)rotation_point = '', 
    plt.show()


#if __name__ == '__main__':
#    greedy_main()