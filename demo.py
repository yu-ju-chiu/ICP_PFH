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

    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])

    # Run ICP
    pc_aligned, errors, ps_list, pt_list = icp(pc_source, pc_target)

    # Plot the original pc
    pc_aligned = utils.convert_matrix_to_pc(pc_aligned.T)
    utils.view_pc([pc_aligned, pc_target], None, ['b', 'r'], ['o', '^'])

    # Plot simplify point cloud
    ps = utils.convert_pc_to_matrix(pc_source)[:, ps_list]
    ps2 = utils.convert_matrix_to_pc(ps)
    pt = utils.convert_pc_to_matrix(pc_target)[:, pt_list]
    pt2 = utils.convert_matrix_to_pc(pt)
    # fig2 = utils.view_pc([ps2, pt2], None, ['g', 'b'], ['x', '<'])
    # draw_lines_3d_numpy(sss, ttt, fig2)

    plt.show()

    # Plot the result
    index = np.arange(1,len(errors)+1,1) 
    plt.plot(index, errors, color='blue')
    plt.xlabel('Iteration')  
    plt.ylabel('Error')  
    plt.show()

if __name__ == '__main__':
    main()
