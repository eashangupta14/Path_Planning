import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import KDTree


# parameter
#N_SAMPLE = 50  # number of sample_points
N_KNN = 100  # number of edge from one sampled point
MAX_EDGE_LEN = 400.0  # [m] Maximum edge length

show_animation = True


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index, orientation = 1000):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index
        self.orientation = orientation

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," +\
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(start_x, start_y, goal_x, goal_y,obstacle_x_list, obstacle_y_list, robot_radius, main_class,num_samples, rng=None):
    """
    Run probabilistic road map planning

    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    :param robot_radius: robot radius
    :param rng: (Optional) Random generator
    :return:
    """
    coll_checker = main_class
    N_SAMPLE = num_samples
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)

    sample_x, sample_y = sample_points(start_x, start_y, goal_x, goal_y,robot_radius,obstacle_x_list, obstacle_y_list,obstacle_kd_tree, rng,N_SAMPLE)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y,robot_radius, obstacle_kd_tree, coll_checker)

    rx, ry = dijkstra_planning(start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y, coll_checker)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree, coll_checker):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True, 0

    D = 25
    n_step = round(d / D)

    # goal point check
    for i in range(n_step):
        #print(i)
        if coll_checker.is_collision(x, y, yaw):
            return True, 0
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    #  goal point check
    if coll_checker.is_collision(gx, gy, yaw):
        return True, 0
        
    return False, yaw  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree, coll_checker):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):

        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            collide,yaw =  is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree, coll_checker)
            if not collide:    
                edge_id.append([indexes[ii],yaw*180/np.pi])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    #  plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y,coll_checker ):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1, orientation = 0)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True
    print('Starting')
    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            print("Final Cost: ",goal_node.cost)
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i][0]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id, orientation = road_map[c_id][i][1])

            if not verify_robot_constraint(current,node, coll_checker):
                continue

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
                    open_set[n_id].orientation = road_map[c_id][i][1]
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry

def verify_robot_constraint(current,node,coll_checker ):
        c_orient = current.orientation
        n_orient = node.orientation
        px = current.x
        py = current.y

        if c_orient < 0:
            c_orient = 360 + c_orient

        if n_orient < 0:
            n_orient = 360 + n_orient

        n_orient_c = 180 + n_orient
        if n_orient_c > 360:
            n_orient_c = n_orient_c - 360               

        init = c_orient
        no = 0    
        while(np.abs(init-n_orient)//20 != 0 and np.abs(init-n_orient_c)//20 != 0):
            init = init + 20
            #print(init)
            if init > 360:
                init = init - 360
            if coll_checker.is_collision(px,py,init):
                no = 1
                print('Breaking')
                break
        if no == 0:
            return True

        init = c_orient 
        while(np.abs(init-n_orient)//20 != 0 and np.abs(init-n_orient_c)//20 != 0):
            init = init - 20
            if init < 0:
                init = 360 - init
            if coll_checker.is_collision(px,py,init):
                no = 1
                return False
                break
        
        return True

def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng, N_SAMPLE):
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        dist, index = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def prm_main(start,end,grid_sz,obstacles,main_class,num_samples, rng=None):
    print(__file__ + " start!!")

    robot_size = 25*np.sqrt(2) # Radius of Curvature

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

    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, main_class, num_samples, rng=rng)

    #assert rx, 'Cannot found path'

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

   