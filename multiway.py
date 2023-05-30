import open3d as o3d
import sys
import json
import numpy as np

def load_point_clouds(amt, voxel_size=0.0):
    pcds = []
    for i in range(1, amt+1):
        col_img = o3d.io.read_image(f"col{str(i)}.jpg")
        dep_img = o3d.io.read_image(f"dep{str(i)}.png")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity = False)
        with open(f'intrinsic{str(i)}.json') as f:
            intrinsic_json = json.load(f)
        phc = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsic_json["intrinsic_matrix"][2], intrinsic_json["intrinsic_matrix"][3], intrinsic_json["intrinsic_matrix"][0], intrinsic_json["intrinsic_matrix"][1])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, phc)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        pcds.append(pcd.voxel_down_sample(voxel_size=voxel_size))
    return pcds

def load_point_clouds_2(amt, voxel_size=0.0):
    pcds = []
    for i in range(1, amt+1):
        col_img = o3d.io.read_image(f"col{str(i)}.jpg")
        dep_img = o3d.io.read_image(f"dep{str(i)}.png")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity = False)
        with open(f'intrinsic{str(i)}.json') as f:
            intrinsic_json = json.load(f)
        phc = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsic_json["intrinsic_matrix"][2], intrinsic_json["intrinsic_matrix"][3], intrinsic_json["intrinsic_matrix"][0], intrinsic_json["intrinsic_matrix"][1])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, phc)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        pcds.append(pcd)
    return pcds

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

def depth_to_pcd(amt, voxel_size):
    pcds_down = load_point_clouds(amt, voxel_size)
    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
    print("Optimizing PoseGraph ...")
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

    pcds = load_point_clouds_2(amt, voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined)
    o3d.visualization.draw_geometries([pcd_combined])

if __name__ == '__main__':
    depth_to_pcd(int(sys.argv[1]), voxel_size = 0.02)