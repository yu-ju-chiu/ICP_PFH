import numpy as np
import pandas as pd
import open3d as o3d
import utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def random_rotation_matrix():
    # Generate a random rotation vector
    rotation_vector = np.random.rand(3)

    # Normalize the rotation vector
    rotation_vector /= np.linalg.norm(rotation_vector)

    # Generate a random rotation matrix using the rotation vector
    rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()

    return rotation_matrix

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

    split_index_f = len(points) * 3 // 3
    split_index_b = len(points) * 0 // 3

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

    df.to_csv(output_file, index=False,header=None)

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
input_pcd = 'data/cat/ism_test_horse.pcd'
output_csv = 'data/cat/horse.csv'
output_csv_0 = 'data/cat/horse_1_all.csv'
output_csv_1 = 'data/cat/horse_2_all.csv'
# input_csv = 'data/cat/cat_2.csv'
output_csv_random = 'data/cat/cat_2_ran.csv'
# output_csv = 'data/cat/cat_2_rot_trad.csv'

# Define rotation matrix and translation vector
random_matrix = random_rotation_matrix()
print("random_matrix", random_matrix)
# cat
rotation_matrix = np.array([[0.8660254, -0.5, 0],
                           [0.5, 0.8660254, 0],
                           [0, 0, 1]])
 # cat_ICP
rotation_matrix = np.array([[ 0.61901649, -0.47188076,  0.62781139],
 [ 0.69604225,  0.69989354, -0.16023177],
 [-0.36379084,  0.53616936,  0.76169459]])

# horse
rotation_matrix = np.array([[ 0.95254632, -0.1961221,   0.23279097],
 [ 0.30331634,  0.54727063, -0.78006029],
 [ 0.0255874,   0.81365286,  0.58078766]])

 # horse_ICP
rotation_matrix = np.array([[ 0.59216217, -0.51876848,  0.61662243],
 [ 0.70389146,  0.70550962, -0.08241955],
 [-0.3922764,   0.482841,    0.78293281]])


# # Check if the matrix is orthogonal
# is_orthogonal = np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.eye(3))
# print("Is orthogonal:", is_orthogonal)

# # Check if the determinant is 1
# det = np.linalg.det(rotation_matrix)
# print("Determinant:", det)

# # Check if transpose is equal to inverse
# is_transpose_equal_inverse = np.allclose(rotation_matrix.T, np.linalg.inv(rotation_matrix))
# print("Transpose is equal to inverse:", is_transpose_equal_inverse)

translation_vector = np.array([1, 0.2, 0.3])
# rotation_matrix = np.eye(3)
# translation_vector = np.array([50, 0, 0])

# # Calculate the inverse transformation
# inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
# inverse_translation_vector = -translation_vector


# rotate_and_translate(input_csv, output_csv, inverse_rotation_matrix, inverse_translation_vector)
pcd_to_csv(input_pcd, output_csv)
num_points_to_pick = 1500
pick_random_points(output_csv, output_csv, num_points_to_pick)
pc_source = utils.load_pc(output_csv)
pc_target = utils.load_pc(output_csv)
utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
slipt_two(output_csv, output_csv_0, output_csv_1)
pc_source = utils.load_pc(output_csv_0)
pc_target = utils.load_pc(output_csv_1)
utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
rotate_and_translate(output_csv_1, output_csv_1, rotation_matrix, translation_vector)
pc_source = utils.load_pc(output_csv_0)
pc_target = utils.load_pc(output_csv_1)
utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
plt.show()
