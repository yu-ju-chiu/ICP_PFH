import numpy as np
import pandas as pd

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
rotation_matrix = np.array([[0.866, -0.5, 0],
                           [0.5, 0.866, 0.5],
                           [0, 0, 0.866]])

translation_vector = np.array([100, 20, 30])

# # Calculate the inverse transformation
# inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
# inverse_translation_vector = -translation_vector


# rotate_and_translate(input_csv, output_csv, inverse_rotation_matrix, inverse_translation_vector)
num_points_to_pick = 1500
pick_random_points(input_csv, output_csv_random, num_points_to_pick)

rotate_and_translate(output_csv_random, output_csv, rotation_matrix, translation_vector)
