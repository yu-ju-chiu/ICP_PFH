import numpy as np
import pandas as pd
import open3d as o3d
import utils
import matplotlib.pyplot as plt

def pcd_to_csv(input_pcd_file, output_csv_file):
    # Read PCD file
    point_cloud = o3d.io.read_point_cloud(input_pcd_file)

    # Extract points as a NumPy array
    points = np.asarray(point_cloud.points)

    # convert to 0.1
    points = points * 1e-2


    # Create a DataFrame
    df = pd.DataFrame(data=points)

    # Save DataFrame to CSV file
    df.to_csv(output_csv_file, index=False, header=None)

def slipt_two(input_csv_file, forward_csv_file, backward_csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file)

    # Extract the point coordinates as a NumPy array
    points = df.values

    split_index_f = len(points) * 2 // 3
    split_index_b = len(points) * 1 // 3

    # Create DataFrames for forward and backward parts
    forward_df = pd.DataFrame(data=points[:split_index_f])
    backward_df = pd.DataFrame(data=points[split_index_b:])

    # Save DataFrames to CSV files
    forward_df.to_csv(forward_csv_file, index=False,header=None)
    backward_df.to_csv(backward_csv_file, index=False, header=None)

def rotate_and_translate(input_file, output_file, rotation_matrix, translation_vector):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Extract the point coordinates as a NumPy array
    points = df.values

    # Apply rotation matrix and translation vector
    rotated_and_translated_points = np.dot(points, rotation_matrix.T) + translation_vector

    # Update DataFrame with new coordinates
    df.loc[:, :] = rotated_and_translated_points

    # Save the result to a new CSV file

    df.to_csv(output_file, index=False)

def pick_random_points(input_file, output_file, num_points_to_pick):
    # Load the CSV file into a DataFrame without header
    df = pd.read_csv(input_file, header=None)

    # Randomly select num_points_to_pick indices
    selected_indices = np.random.choice(len(df), num_points_to_pick, replace=False)
    selected_indices = np.sort(selected_indices)

    # Extract selected points
    selected_points = df.iloc[selected_indices]

    # Save the selected points to a new CSV file without header
    selected_points.to_csv(output_file, index=False, header=False)
# Example usage:
input_pcd = 'data/cat/ism_test_cat.pcd'
output_pcd = 'data/cat/cat_0.csv'
output_csv_random = 'data/cat/cat_0_ran.csv'
output_csv_f  = 'data/cat/cat_1_f.csv'
output_csv_b  = 'data/cat/cat_2_b.csv'
output_csv_b_inv  = 'data/cat/cat_2_b_inv.csv'
# input_csv = 'data/cat/cat_2.csv'

# input_csv  = 'data/cat/cat_0.csv'
# output_csv_f  = 'data/cat/cat_1_f.csv'
# output_csv_b  = 'data/cat/cat_2_b.csv'
# output_csv_random = 'data/cat/cat_2_ran.csv'
# output_csv = 'data/cat/cat_2_rot_trad.csv'

# Define rotation matrix and translation vector
rotation_matrix = np.array([[0.866, -0.5, 0],
                           [0.5, 0.866, 0],
                           [0, 0, 1]])

translation_vector = np.array([0.01, 0.01, 0.05])

# inverse_rotation_matrix = np.array([[ 0.98866494, -0.02600738, -0.14786903],
#                                 [ 0.01414802,  0.99663842, -0.08069508],
#                                 [0.14947062,  0.07768834,  0.98570942]])

# inverse_translation_vector = np.array([[0.13584385],[0.04252106],[0.03461734]])


# # Calculate the inverse transformation
# inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
# inverse_translation_vector = -translation_vector


rotate_and_translate(output_csv_b, output_csv_b_inv, rotation_matrix, translation_vector)

# pcd_to_csv(input_pcd, output_pcd)
# num_points_to_pick = 1200
# pick_random_points(output_pcd, output_csv_random, num_points_to_pick)
# slipt_two(output_csv_random, output_csv_f, output_csv_b)

# rotate_and_translate(output_csv_b, output_csv_b, rotation_matrix, translation_vector)
pc_source = utils.load_pc(output_csv_f)
# pc_target = utils.load_pc('cat_2_ran.csv')
pc_target = utils.load_pc(output_csv_b_inv)
# pc_target = utils.load_pc('bunny_2_500.csv') # Change this to load in a different target

utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
plt.show()
