import open3d as o3d
import json
import numpy as np
import imageio
import copy

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
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

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def depth_to_point_cloud(): 
    
    # Load JSON files 
    with open('intrinsic1.json') as f:
        intrinsic_json_1 = json.load(f)
    with open('intrinsic2.json') as f2:
        intrinsic_json_2 = json.load(f2)
    # with open('extrinsic.json') as f3:
    #     extrinsic_json = json.load(f3)

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
    pcds = []
    pcds.append(pcd)
    pcds.append(pcd2)
    # for pc in pcds:
    #     pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #     pc.orient_normals_consistent_tangent_plane(100)

    # Create point clouds array of downsampled versions
    # pcds_down = []
    voxel_size = 0.02
    # pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    # pcd_down2 = pcd2.voxel_down_sample(voxel_size=voxel_size)
    # pcds_down.append(pcd_down)
    # pcds_down.append(pcd_down2)

    source_down, source_fpfh = preprocess_point_cloud(pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    # result_icp = refine_registration(pcd, pcd2, source_fpfh, target_fpfh,
    #                              voxel_size, result_ransac)
    print(result_ransac)
    draw_registration_result(pcd, pcd2, result_ransac.transformation)

    # Estimate normals
    # for pc in pcds_down:
    #     pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #     pc.orient_normals_consistent_tangent_plane(100)

    # Execute registration
    # max_correspondence_distance_coarse = voxel_size * 15
    # max_correspondence_distance_fine = voxel_size * 1.5

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     pose_graph = full_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine)

    # Optimize registration
    # option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=max_correspondence_distance_fine, edge_prune_threshold=0.25, reference_node=0)
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     o3d.pipelines.registration.global_optimization(pose_graph,
    #     o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    #     o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    #     option)

    # Transform and visualize
    # for point_id in range(len(pcds)):
    #     print(pose_graph.nodes[point_id].pose)
    #     pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    # o3d.visualization.draw_geometries(pcds)

    # Scale point clouds
    # pcd.scale(1000.0, center=(0, 0, 0))
    # pcd2.scale(1000.0, center=(0, 0, 0))

    # Get extrinsic matrices
    # R = np.array([extrinsic_json["rotation_matrix"][0:3], extrinsic_json["rotation_matrix"][3:6], extrinsic_json["rotation_matrix"][6:9]])
    # T = np.array([[x] for x in extrinsic_json["translation_matrix"][0:3]])

    # Create homogeneous translation matrix
    # t_homo = np.vstack((np.hstack((np.eye(3), T)), [0, 0, 0, 1]))

    # Create homogeneous rotation matrix
    # R_homo = np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1]))

    # Create homogeneous transformation matrix
    # homo = t_homo@R_homo
    # homo_inv = np.linalg.inv(homo) # Take inverse of homogeneous transformation matrix
    # print(t_homo, '\n')
    # print(R_homo, '\n')
    # print(homo, '\n')
    # print(homo_inv, '\n')

    # reg_p2p = open3d.pipelines.registration.registration_icp(pcd2, pcd, 1, homo_mat, open3d.pipelines.registration.TransformationEstimationPointToPoint())
    # pcd2.transform(homo)
    # pcd.translate(T)
    # pcd += pcd2
    # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # pcd.orient_normals_consistent_tangent_plane(100)
    # mesh, _ = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, 9)
    # open3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    depth_to_point_cloud()