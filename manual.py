import open3d as o3d
import copy
import imageio
import numpy as np
import json

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # User picks points
    vis.destroy_window()
    return vis.get_picked_points()


def demo_manual_registration():

    # Load image data and create point clouds
    # test_img = imageio.imread("color0.jpg")
    # height, width = test_img.shape[:2]

    # pcds = []
    # for i in range(0, 2):
    #     col_img = o3d.io.read_image(f"color{str(i)}.jpg")
    #     dep_img = o3d.io.read_image(f"depth{str(i)}.png")
    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity = False)
    #     with open(f'intrinsic{str(i)}.json') as f:
    #         intrinsic_json = json.load(f)
    #     phc = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_json["intrinsic_matrix"][2], intrinsic_json["intrinsic_matrix"][3], intrinsic_json["intrinsic_matrix"][0], intrinsic_json["intrinsic_matrix"][1])
    #     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, phc)
    #     pcds.append(pcd)
    source = o3d.io.read_point_cloud("bekertje3.ply")
    target = o3d.io.read_point_cloud("bekertje4.ply")
    draw_registration_result(source, target, np.identity(4))

    # Pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # Estimate rough transformation using correspondences
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # Point-to-point ICP for refinement
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)

if __name__ == "__main__":
    demo_manual_registration()