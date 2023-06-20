import open3d as o3d
import sys
import json
import numpy as np
import imageio

def load_point_clouds(amt, voxel_size=0.0):
    """
    Load image data and create point clouds
    """

    # test_img = imageio.imread("color0.jpg")
    # height, width = test_img.shape[:2]

    # pcds = []
    # for i in range(0, amt):
    #     col_img = o3d.io.read_image(f"color{str(i)}.jpg")
    #     dep_img = o3d.io.read_image(f"depth{str(i)}.png")
    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity = False)
    #     with open(f'intrinsic{str(i)}.json') as f:
    #         intrinsic_json = json.load(f)
    #     phc = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_json["intrinsic_matrix"][2], intrinsic_json["intrinsic_matrix"][3], intrinsic_json["intrinsic_matrix"][0], intrinsic_json["intrinsic_matrix"][1])
    #     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, phc)
    #     pcd.estimate_normals()
    #     pcd.orient_normals_consistent_tangent_plane(100)
    #     pcds.append(pcd)
    source = o3d.io.read_point_cloud("bekertje3.ply")
    source.estimate_normals()
    source.orient_normals_consistent_tangent_plane(100)
    target = o3d.io.read_point_cloud("bekertje4.ply")
    target.estimate_normals()
    target.orient_normals_consistent_tangent_plane(100)
    return [source, target]

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
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
            if target_id == source_id + 1:  # Odometry case
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
            else:  # Loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def depth_to_pcd(amt, voxel_size):
    pcds = load_point_clouds(amt, voxel_size)
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    # Execute multiway registration
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
        
    # Optimize registration
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    # Visualize
    pcds = load_point_clouds(amt, voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    o3d.visualization.draw_geometries([pcd_combined])

if __name__ == '__main__':
    depth_to_pcd(int(sys.argv[1]), voxel_size = 0.02)