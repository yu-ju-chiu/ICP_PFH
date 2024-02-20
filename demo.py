#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
from ICP import icp

def main():
    # Import the cloud
    # # mug
    # pc_source = utils.load_pc('data/mug/cloud_icp_source.csv')
    # pc_target = utils.load_pc('data/mug/cloud_icp_target3.csv')
    # # bunny
    # pc_source = utils.load_pc('data/bunny/bunny_0_500.csv')
    # pc_target = utils.load_pc('data/bunny/bunny_1_500.csv')
    # # cat
    pc_source = utils.load_pc('data/cat/cat_2_rot_trad.csv')
    pc_target = utils.load_pc('data/cat/cat_1.csv')
    # # horse
    # pc_source = utils.load_pc('data/cat/horse_2.csv')
    # pc_target = utils.load_pc('data/cat/horse_1.csv')

    fig1 = utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title("Cat point clouds initial")

    # Run ICP
    pc_aligned, errors, ps_list, pt_list, sss, ttt = icp(pc_source, pc_target)
    draw_lines_3d_numpy(sss, ttt, fig1)

    # Plot the original pc
    pc_aligned = utils.convert_matrix_to_pc(pc_aligned.T)
    utils.view_pc([pc_aligned, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title("Cat point clouds after ICP-PFH alignment")

    # Plot simplify point cloud
    ps = utils.convert_pc_to_matrix(pc_source)[:, ps_list]
    ps2 = utils.convert_matrix_to_pc(ps)
    pt = utils.convert_pc_to_matrix(pc_target)[:, pt_list]
    pt2 = utils.convert_matrix_to_pc(pt)

    plt.show()

    # # Plot the result
    # index = np.arange(1,len(errors)+1,1) 
    # plt.plot(index, errors, color='blue')
    # plt.xlabel('Iteration')  
    # plt.ylabel('Error')  
    # plt.show()

def draw_lines_3d_numpy(array1, array2, fig):
    # Extract x, y, z coordinates from the NumPy arrays
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    x1, y1, z1 = array1[:, 0], array1[:, 1], array1[:, 2]
    x2, y2, z2 = array2[:, 0], array2[:, 1], array2[:, 2]

    # Create a 3D plot
    ax = fig.gca()
    # fig.add_subplot(111, projection='3d')

    for x1_point, y1_point, z1_point, x2_point, y2_point, z2_point in zip(x1, y1, z1, x2, y2, z2):
        ax.plot([x1_point, x2_point], [y1_point, y2_point], [z1_point, z2_point], color='gray')

    plt.draw()
    return fig

def draw_lines_3d_numpy(array1, array2, fig):
    # Extract x, y, z coordinates from the NumPy arrays
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    x1, y1, z1 = array1[:, 0], array1[:, 1], array1[:, 2]
    x2, y2, z2 = array2[:, 0], array2[:, 1], array2[:, 2]

    # Create a 3D plot
    ax = fig.gca()
    # fig.add_subplot(111, projection='3d')

    for x1_point, y1_point, z1_point, x2_point, y2_point, z2_point in zip(x1, y1, z1, x2, y2, z2):
        ax.plot([x1_point, x2_point], [y1_point, y2_point], [z1_point, z2_point], color='gray')

    plt.draw()
    return fig

if __name__ == '__main__':
    main()
