from open3d import *
import json
import numpy as np

def depth_to_point_cloud(): 
    with open('intrinsic1.json') as f:
        intrinsic_json_1 = json.load(f)
    with open('intrinsic2.json') as f2:
        intrinsic_json_2 = json.load(f2)
    with open('extrinsic.json') as f3:
        extrinsic_json = json.load(f3)
    col_img_1 = open3d.io.read_image("color1.jpg")
    dep_img_1 = open3d.io.read_image("depth1.png")
    col_img_2 = open3d.io.read_image("color2.jpg")
    dep_img_2 = open3d.io.read_image("depth2.png")
    rgbd1 = open3d.geometry.RGBDImage.create_from_color_and_depth(col_img_1, dep_img_1, convert_rgb_to_intensity = False)
    rgbd2 = open3d.geometry.RGBDImage.create_from_color_and_depth(col_img_2, dep_img_2, convert_rgb_to_intensity = False)
    phc = open3d.camera.PinholeCameraIntrinsic()
    phc2 = open3d.camera.PinholeCameraIntrinsic()
    phc.intrinsic_matrix = [intrinsic_json_1["intrinsic_matrix"][2], 0, intrinsic_json_1["intrinsic_matrix"][0]], [0, intrinsic_json_1["intrinsic_matrix"][3], intrinsic_json_1["intrinsic_matrix"][1]], [0, 0, 1]
    phc2.intrinsic_matrix = [intrinsic_json_2["intrinsic_matrix"][2], 0, intrinsic_json_2["intrinsic_matrix"][0]], [0, intrinsic_json_2["intrinsic_matrix"][3], intrinsic_json_2["intrinsic_matrix"][1]], [0, 0, 1]
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, phc)
    pcd2 = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, phc2)
    rot_mat = np.array([extrinsic_json["rotation_matrix"][0:3], extrinsic_json["rotation_matrix"][3:6], extrinsic_json["rotation_matrix"][6:9]])
    trans_list = [(x / 1000)/2.4 for x in extrinsic_json["translation_matrix"][0:3]]
    pcd2.translate(trans_list)
    pcd2.rotate(rot_mat)
    pcd += pcd2
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    open3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    depth_to_point_cloud()