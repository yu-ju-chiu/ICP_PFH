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

    # Create a DataFrame
    df = pd.DataFrame(data=points)

    # Save DataFrame to CSV file
    df.to_csv(output_csv_file, index=False)

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
    forward_df.to_csv(forward_csv_file, index=False)
    backward_df.to_csv(backward_csv_file, index=False)

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

    # Extract selected points
    selected_points = df.iloc[selected_indices]

    # Save the selected points to a new CSV file without header
    selected_points.to_csv(output_file, index=False, header=False)
# Example usage:
input_csv = 'data/cat/cat_2.csv'
output_csv_random = 'data/cat/cat_2_ran.csv'
output_csv = 'data/cat/cat_2_rot_trad.csv'

# Define rotation matrix and translation vector
rotation_matrix = np.array([[0.8660254, -0.5, 0],
                           [0.5, 0.8660254, 0],
                           [0, 0, 1]])

# Check if the matrix is orthogonal
is_orthogonal = np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.eye(3))
print("Is orthogonal:", is_orthogonal)

# Check if the determinant is 1
det = np.linalg.det(rotation_matrix)
print("Determinant:", det)

# Check if transpose is equal to inverse
is_transpose_equal_inverse = np.allclose(rotation_matrix.T, np.linalg.inv(rotation_matrix))
print("Transpose is equal to inverse:", is_transpose_equal_inverse)

translation_vector = np.array([100, 20, 30])
# rotation_matrix = np.eye(3)
# translation_vector = np.array([50, 0, 0])

# # Calculate the inverse transformation
# inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
# inverse_translation_vector = -translation_vector


# rotate_and_translate(input_csv, output_csv, inverse_rotation_matrix, inverse_translation_vector)
num_points_to_pick = 1500
# pick_random_points(input_csv, output_csv_random, num_points_to_pick)

rotate_and_translate(input_csv, output_csv, rotation_matrix, translation_vector)
