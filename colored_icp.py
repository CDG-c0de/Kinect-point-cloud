import open3d as o3d
import json
import numpy as np
import imageio
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def depth_to_point_cloud():

    # # Load image data and create point clouds
    test_img = imageio.imread("color0.jpg")
    height, width = test_img.shape[:2]
    
    pcds = []
    for i in range(0, 2):
        col_img = o3d.io.read_image(f"color{str(i)}.jpg")
        dep_img = o3d.io.read_image(f"depth{str(i)}.png")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity = False)
        with open(f'intrinsic{str(i)}.json') as f:
            intrinsic_json = json.load(f)
        phc = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_json["intrinsic_matrix"][2], intrinsic_json["intrinsic_matrix"][3], intrinsic_json["intrinsic_matrix"][0], intrinsic_json["intrinsic_matrix"][1])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, phc)
        pcds.append(pcd)

    pcd, pcd2 = pcds

    voxel_radius = [0.08, 0.04, 0.02, 0.01]
    max_iter = [100, 50, 30, 14]
    current_transformation = np.identity(4)
    for scale in range(4):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        # Downsample point clouds with given radii
        source_down = pcd2.voxel_down_sample(radius)
        target_down = pcd.voxel_down_sample(radius)

        # Estimate normals
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

    threshold = 10
    
    # Apply point to point ICP and visualize
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2, pcd, threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(pcd2, pcd, reg_p2p.transformation)

if __name__ == '__main__':
    depth_to_point_cloud()