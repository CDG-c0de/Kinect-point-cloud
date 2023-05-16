from open3d import *
import json
import numpy as np
import imageio

def swap_yaw_pitch(matrix):
    new_matrix = np.copy(matrix)
    new_matrix[0:3, 0:3] = matrix[0:3, [1, 0, 2]]
    new_matrix[0:3, [1, 0, 2]] = matrix[0:3, 0:3]
    return new_matrix

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
    col_img_1 = open3d.io.read_image("color1.jpg")
    dep_img_1 = open3d.io.read_image("depth1.png")
    col_img_2 = open3d.io.read_image("color2.jpg")
    dep_img_2 = open3d.io.read_image("depth2.png")

    # Create RGBD images
    rgbd1 = open3d.geometry.RGBDImage.create_from_color_and_depth(col_img_1, dep_img_1, convert_rgb_to_intensity = False)
    rgbd2 = open3d.geometry.RGBDImage.create_from_color_and_depth(col_img_2, dep_img_2, convert_rgb_to_intensity = False)

    # Create pinhole cameras
    phc = open3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_json_1["intrinsic_matrix"][2], intrinsic_json_1["intrinsic_matrix"][3], intrinsic_json_1["intrinsic_matrix"][0], intrinsic_json_1["intrinsic_matrix"][1])
    phc2 = open3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_json_2["intrinsic_matrix"][2], intrinsic_json_2["intrinsic_matrix"][3], intrinsic_json_2["intrinsic_matrix"][0], intrinsic_json_2["intrinsic_matrix"][1])
    
    # Create point clouds
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, phc)
    pcd2 = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, phc2)

    # Get extrinsic matrices
    R = np.array([extrinsic_json["rotation_matrix"][0:3], extrinsic_json["rotation_matrix"][3:6], extrinsic_json["rotation_matrix"][6:9]])
    T = np.array([[x/1000/24] for x in extrinsic_json["translation_matrix"][0:3]])

    # Create homogeneous transformation matrix
    t_homo = np.vstack((np.hstack((np.eye(3), T)), [0, 0, 0, 1]))
    R_homo = np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1]))
    homo = t_homo@R_homo
    homo_inv = np.linalg.inv(homo) # Take inverse of homogeneous transformation matrix
    print(t_homo, '\n')
    print(R_homo, '\n')
    print(homo, '\n')
    print(homo_inv, '\n')
    print(swap_yaw_pitch(homo_inv))

    fix = np.vstack((np.hstack((np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), [[0], [0], [0]])), [0, 0, 0, 1]))

    # reg_p2p = open3d.pipelines.registration.registration_icp(pcd2, pcd, 1, homo_mat, open3d.pipelines.registration.TransformationEstimationPointToPoint())
    pcd2.transform(swap_yaw_pitch(homo_inv))
    # pcd2.transform(fix)
    pcd += pcd2
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # pcd.orient_normals_consistent_tangent_plane(100)
    # mesh, _ = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, 9)
    open3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    depth_to_point_cloud()