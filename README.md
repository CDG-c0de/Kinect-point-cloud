# Kinect-point-cloud
This program generates and combines point clouds from the data aquired from the Azure Kinect cameras.<br />
It does not save the point cloud data, only displays it live.
## Dependencies
The following pip packages are required: <br />
Open3D <br />
Numpy <br />
JSON
## How to run
Simply execute the python script like: python program.py
## Input
The program requires the following files to be present in the root directory, with the same names: <br /> <br />
intrinsic1.json&nbsp;&nbsp;&nbsp;&nbsp;*the camera intrinsic matrix of the master camera* <br /><br />
intrinsic2.json&nbsp;&nbsp;&nbsp;&nbsp;*the camera intrinsic matrix of the subordinate camera* <br /><br />
extrinsic.json&nbsp;&nbsp;&nbsp;&nbsp;*the camera extrinsic matrices, for the transformation fom sub camera to master camera* <br /><br />
color1.jpg&nbsp;&nbsp;&nbsp;&nbsp;*the color image of the master camera* <br /><br />
color2.jpg&nbsp;&nbsp;&nbsp;&nbsp;*the color image of the subordinate camera* <br /><br />
depth1.png&nbsp;&nbsp;&nbsp;&nbsp;*the depth image of the master camera* <br /><br />
depth2.png&nbsp;&nbsp;&nbsp;*the depth image of the subordinate camera*
