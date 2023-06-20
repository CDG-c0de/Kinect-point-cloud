import open3d as o3d
import numpy as np
import sys

def run(file):
    pcd = o3d.io.read_point_cloud(file)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.025)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

if __name__ == "__main__":
    run(sys.argv[1])