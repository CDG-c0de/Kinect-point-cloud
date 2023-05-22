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
    with open('extrinsic.json') as f3:
        extrinsic_json = json.load(f3)

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

    # Scale point clouds to match OpenCV units
    pcd.scale(1000.0, center=(0, 0, 0))
    pcd2.scale(1000.0, center=(0, 0, 0))

    # Get extrinsic matrices
    R = np.array([extrinsic_json["rotation_matrix"][0:3], extrinsic_json["rotation_matrix"][3:6], extrinsic_json["rotation_matrix"][6:9]])
    T = np.array([[x] for x in extrinsic_json["translation_matrix"][0:3]])

    # Create homogeneous translation matrix
    t_homo = np.vstack((np.hstack((np.eye(3), T)), [0, 0, 0, 1]))

    # Create homogeneous rotation matrix
    R_homo = np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1]))

    # Create homogeneous transformation matrix
    homo = t_homo@R_homo

    # Execute transformation and merge point clouds
    pcd2.transform(homo)
    pcd += pcd2

    # Visualize result
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    depth_to_point_cloud()