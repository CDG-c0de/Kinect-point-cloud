import open3d as o3d
import json
import imageio
import numpy as np

with open('intrinsic0.json') as f:
    intrinsic_json_1 = json.load(f)

# Get dimensions of images
test_img = imageio.imread("color0.jpg")
height, width = test_img.shape[:2]

# Load color and depth images
col_img_1 = o3d.io.read_image("color0.jpg")
dep_img_1 = o3d.io.read_image("depth0.png")

# Create RGBD images
rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img_1, dep_img_1, convert_rgb_to_intensity = False)

# Create pinhole cameras
phc = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_json_1["intrinsic_matrix"][2], intrinsic_json_1["intrinsic_matrix"][3], intrinsic_json_1["intrinsic_matrix"][0], intrinsic_json_1["intrinsic_matrix"][1])

# Create point clouds
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, phc)

# Estimate and orient normals
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    
vertices_to_remove = densities < np.quantile(densities, 0.1)
mesh.remove_vertices_by_mask(vertices_to_remove)
# Visualize
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)