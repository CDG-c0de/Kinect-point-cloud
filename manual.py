import open3d as o3d
import copy
import numpy as np
import json

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def demo_manual_registration():
    print("Demo for manual ICP")
    pcds = []
    for i in range(1, 3):
        col_img = o3d.io.read_image(f"color{str(i)}.jpg")
        dep_img = o3d.io.read_image(f"depth{str(i)}.png")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity = False)
        with open(f'intrinsic{str(i)}.json') as f:
            intrinsic_json = json.load(f)
        phc = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsic_json["intrinsic_matrix"][2], intrinsic_json["intrinsic_matrix"][3], intrinsic_json["intrinsic_matrix"][0], intrinsic_json["intrinsic_matrix"][1])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, phc)
        pcds.append(pcd)
    source = pcds[0]
    target = pcds[1]
    source = source.voxel_down_sample(0.001)
    target = target.voxel_down_sample(0.001)
    # pcd_data = o3d.data.DemoICPPointClouds()
    # source = o3d.io.read_point_cloud(pcd_data.paths[0])
    # target = o3d.io.read_point_cloud(pcd_data.paths[2])
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")


if __name__ == "__main__":
    demo_manual_registration()