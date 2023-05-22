import open3d as o3d
import json
import numpy as np
import imageio

def depth_to_point_cloud(): 
    
    # Load JSON files 
    with open('intrinsic1.json') as f:
        intrinsic_json_1 = json.load(f)
    with open('intrinsic2.json') as f2:
        intrinsic_json_2 = json.load(f2)

    # Get dimensions of images
    test_img = imageio.v2.imread("color1.jpg")
    height, width = test_img.shape[:2]

    # Load color and depth images
    col_img_1 = o3d.io.read_image("color1.jpg")
    dep_img_1 = o3d.io.read_image("depth1.png")
    col_img_2 = o3d.io.read_image("color2.jpg")
    dep_img_2 = o3d.io.read_image("depth2.png")

    # Create RGBD images
    rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img_1, dep_img_1, convert_rgb_to_intensity = False)
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img_2, dep_img_2, convert_rgb_to_intensity = False)

    # Create pinhole cameras
    phc = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_json_1["intrinsic_matrix"][2], intrinsic_json_1["intrinsic_matrix"][3], intrinsic_json_1["intrinsic_matrix"][0], intrinsic_json_1["intrinsic_matrix"][1])
    phc2 = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_json_2["intrinsic_matrix"][2], intrinsic_json_2["intrinsic_matrix"][3], intrinsic_json_2["intrinsic_matrix"][0], intrinsic_json_2["intrinsic_matrix"][1])
    
    # Create point clouds
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, phc)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, phc2)

    # Execute ICP registration
    current_transformation = np.identity(4)
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        # Downsample point clouds with given radii
        source_down = pcd2.voxel_down_sample(radius)
        target_down = pcd.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # Apply colored ICP registration
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation

    # Apply transformation and merge point clouds
    pcd += pcd2.transform(current_transformation)

    # Visualize result
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    depth_to_point_cloud()