import open3d as o3d
import json
import imageio

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def depth_to_point_cloud(): 

    # Load image data and create point clouds
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

    voxel_size = 0.05

    # Downsample point clouds and compute FPFH features
    source_down, source_fpfh = preprocess_point_cloud(pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)
    
    # Execute global registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    pcd2.transform(result_ransac.transformation)

    # Combine point clouds
    pcd += pcd2

    # Visualize result
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    depth_to_point_cloud()