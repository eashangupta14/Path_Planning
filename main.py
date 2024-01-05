# This is the main file for HW 4

import numpy as np
import argparse
from utils import pathplanner


def planner():
	myclass = pathplanner()
	if opt.qnum == 1:
		myclass.plot_cs(theta = opt.theta)
	elif opt.qnum == 2:
		myclass.greedy(opt.start,opt.des,opt.grid_sz)
	elif opt.qnum == 3:
		myclass.safe(opt.start,opt.des,opt.grid_sz)
	elif opt.qnum == 4:
		myclass.PRM(opt.start,opt.des,opt.grid_sz, opt.num_samples)
	elif opt.qnum == 5:
		myclass.RRT(opt.start,opt.des,opt.grid_sz, opt.num_samples)
		




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--qnum', type=int, default=1, help='Question Number')
    parser.add_argument('--theta', type = float, default = 0)
    parser.add_argument('--grid_sz', type = int, default = 2)
    parser.add_argument('--num_samples', type = int, default = 50)
    parser.add_argument('--start', nargs='+', type = int)
    parser.add_argument('--des', nargs='+', type = int)


    '''
    start = [50,300-50]
	destination = [750,300-50]
    parser.add_argument('--start_time', type = int, default = 0)
    parser.add_argument('--end_time', type = int, default = 30)
    
    parser.add_argument('--y_init', nargs='+', type = float)
    parser.add_argument('--size', nargs='+', type = int)
    parser.add_argument('--num_components', default = 64)
    parser.add_argument('--method', default = 'knn')
    parser.add_argument('--root', default = './Data')
    parser.add_argument('--k_neighbours', default = 6)
    parser.add_argument('--do_pca', action = 'store_true')
    parser.add_argument('--do_lda', action = 'store_true')
    parser.add_argument('--visualize', action = 'store_true')
	'''


    opt = parser.parse_args()
    
    planner()