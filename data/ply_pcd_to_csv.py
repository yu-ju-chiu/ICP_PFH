import open3d as o3d
import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import csv


def pcd_to_csv(pcd_file, csv_file):
    # Read PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Extract points from the PointCloud object
    points = pd.DataFrame(pcd.points, columns=['x', 'y', 'z'])

    # Save points to CSV file
    points.to_csv(csv_file, index=False, header=False)

def ply_to_csv(input_ply_file, output_csv_file):
    # Read the PLY file
    ply_data = PlyData.read(input_ply_file)

    # Extract x, y, and z coordinates from the 'vertex' element
    x = ply_data['vertex']['x']
    y = ply_data['vertex']['y']
    z = ply_data['vertex']['z']

    # Combine coordinates into a list of tuples
    coordinates = list(zip(x, y, z))

    # Write data to CSV file
    with open(output_csv_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(['x', 'y', 'z'])
        # Write coordinates
        csv_writer.writerows(coordinates)

if __name__ == "__main__":
    # pcd to csv
    pcd_file_path = "data/object_template_0.pcd"
    csv_file_path = "csv/rabbit_.csv"

    pcd_to_csv(pcd_file_path, csv_file_path)
    utils.view_pc([utils.load_pc(csv_file_path)], None, ['b'], ['o'])
    plt.show()


    # ply to csv
    ply_file_path = "data/bun315.ply"
    csv_file_path = "bunny_1.csv"

    ply_to_csv(ply_file_path, csv_file_path)
    utils.view_pc([utils.load_pc(csv_file_path)], None, ['b'], ['o'])
    plt.show()
    