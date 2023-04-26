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

class RecorderWithCallback:

    def __init__(self, config, device, filename, align_depth_to_color):
        # Global flags
        self.flag_exit = False
        self.flag_record = False
        self.filename = filename

        self.align_depth_to_color = align_depth_to_color
        self.recorder = open3d.io.AzureKinectRecorder(config, device)
        if not self.recorder.init_sensor():
            raise RuntimeError('Failed to connect to sensor')

    def escape_callback(self, vis):
        self.flag_exit = True
        if self.recorder.is_record_created():
            print('Recording finished.')
        else:
            print('Nothing has been recorded.')
        return False

    def space_callback(self, vis):
        if self.flag_record:
            print('Recording paused. '
                  'Press [Space] to continue. '
                  'Press [ESC] to save and exit.')
            self.flag_record = False

        elif not self.recorder.is_record_created():
            if self.recorder.open_record(self.filename):
                print('Recording started. '
                      'Press [SPACE] to pause. '
                      'Press [ESC] to save and exit.')
                self.flag_record = True

        else:
            print('Recording resumed, video may be discontinuous. '
                  'Press [SPACE] to pause. '
                  'Press [ESC] to save and exit.')
            self.flag_record = True

        return False

    def run(self):
        glfw_key_escape = 256
        glfw_key_space = 32
        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.register_key_callback(glfw_key_space, self.space_callback)

        vis.create_window('recorder', 1920, 540)
        print("Recorder initialized. Press [SPACE] to start. "
              "Press [ESC] to save and exit.")

        vis_geometry_added = False
        while not self.flag_exit:
            rgbd = self.recorder.record_frame(self.flag_record,
                                              self.align_depth_to_color)
            if rgbd is None:
                continue

            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()

        self.recorder.close_record()

if __name__ == '__main__':
    # config = open3d.io.AzureKinectSensorConfig()
    # filename = '{date:%Y-%m-%d-%H-%M-%S}.mkv'.format(
    #     date=datetime.datetime.now())
    # print('Prepare writing to {}'.format(filename))

    # device = 0
    # if device < 0 or device > 255:
    #     print('Unsupported device id, fall back to 0')
    #     device = 0

    # r = RecorderWithCallback(config, device, filename,
    #                          False)
    # r.run()
    # azure_kinect_mkv_reader.main(filename, os.getcwd() + '/frames')
    # os.remove(filename)
    depth_to_point_cloud()