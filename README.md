# Kinect-point-cloud
This repo consists of multiple Python scripts; ICP registration, stereo calibration, multiway registration, estimating normals, manual registration, global registration, mesh generation, and colored point cloud registration.
## Dependencies
The required pip packages can be installed using the requirements.txt file: `pip install -r requirements.txt`
## How to run
`colored_icp.py`: run like `python colored_icp.py`, the script will automatically take 2 jpg images and their corresponding png depth images, along with their intrinsic parameters. <br />
`create_mesh_from_pc_poisson.py`: run like `python create_mesh_from_pc_poisson.py <filename>`, provide the script with a file containing point cloud information, like a pcd or ply file. <br />
`global_reg.py`: run like `python global_reg.py`, the script will automatically take 2 jpg images and their corresponding png depth images, along with their intrinsic parameters. <br />
`manual.py`: run like `python manual.py`, the script will automatically take 2 jpg images and their corresponding png depth images, along with their intrinsic parameters. <br />
`multiway.py`: run like `python multiway.py <amount>`, provide the number of pcds you have, the script will automatically take the proper amount of jpg images and their corresponding png depth images, along with their intrinsic parameters. <br />
`normals.py`: run like `python normals.py`, the script will automatically take 1 jpg image and its corresponding png depth image, along with its intrinsic parameters. <br />
`stereo_calib.py`: run like `python stereo_calib.py`, the script will automatically take 2 jpg images and their corresponding png depth images, along with their intrinsic and extrinsic parameters. <br />
## Input formatting
The scripts that automatically take input files assume the following format, where index represents the camera index (index is 0 up to amount of cameras). This formatting is already correct if the code from [C++ repo](https://github.com/CDG-c0de/Kinect-calib-and-capture) is used: <br /> <br />
intrinsic[index].json&nbsp;&nbsp;&nbsp;&nbsp;*the camera intrinsic matrix of camera [index]* <br /><br />
color[index].jpg&nbsp;&nbsp;&nbsp;&nbsp;*the color image of camera [index]* <br /><br />
depth[index].png&nbsp;&nbsp;&nbsp;&nbsp;*the depth image of camera [index]* <br /><br />

The stereo calibration script also requires extrinsics:
extrinsic.json&nbsp;&nbsp;&nbsp;&nbsp;*the camera extrinsic matrices, for the transformation from sub camera to master camera* <br /><br />
