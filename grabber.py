import json
import open3d as o3d
import pykinect_azure as pykinect
import numpy as np
import cv2
import copy

# Set depth sensor delay to 160 micro seconds, to prevent interference
MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC = 160

def default_config():
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
    device_config.subordinate_delay_off_master_usec = 0
    device_config.synchronized_images_only = True
    return device_config

if __name__ == "__main__":

    # Import k4a DLL
    pykinect.initialize_libraries("C:\\Program Files\\Azure Kinect SDK v1.4.0\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll")

    master_config = default_config()
    master_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
    master_config.depth_delay_off_color_usec = -(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC//2)

    sub_config = default_config()
    sub_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
    sub_config.depth_delay_off_color_usec = (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC//2)

    amt_devs = pykinect.Device.device_get_installed_count()
    print(amt_devs)

    devs = []
    for i in list(reversed(range(amt_devs))):
        devs.append(pykinect.start_device(i, sub_config))

    captures = []
    pcds = []
    while 1:
        captures = []
        # Get capture
        for i in range(len(devs)):
            captures.append(devs[i].update())

        # Get the color image from the capture
        rets = [None] * len(captures)
        color_images = [None] * len(captures)
        point_rets = [None] * len(captures)
        points_list = [None] * len(captures)
        print(len(captures))
        for i in range(len(captures)):
            rets[i], color_images[i] = captures[i].get_color_image()
            point_rets[i], points_list[i] = captures[i].get_transformed_pointcloud()
        if all(i for i in rets)  and all(j for j in point_rets):
            for i in range(len(color_images)):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_list[i])
                colors = cv2.cvtColor(color_images[i], cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcds.append(copy.deepcopy(pcd))
            break
    for i in range(len(captures)):
        o3d.io.write_point_cloud(f"testing{i}.pcd", pcds[i])