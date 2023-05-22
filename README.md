# Kinect-point-cloud
This repo consists of two python scripts, one uses ICP registration, the other applies the extrinsic matrices acquired from an external source.
## Dependencies
The required pip packages can be installed using the requirements.txt file: `pip install -r requirements.txt`
## How to run
Simply execute the python script like: python program.py
## Input
The script using the ICP method requires the following files to be present in the root directory, with the same names: <br /> <br />
intrinsic1.json&nbsp;&nbsp;&nbsp;&nbsp;*the camera intrinsic matrix of the master camera* <br /><br />
intrinsic2.json&nbsp;&nbsp;&nbsp;&nbsp;*the camera intrinsic matrix of the subordinate camera* <br /><br />
color1.jpg&nbsp;&nbsp;&nbsp;&nbsp;*the color image of the master camera* <br /><br />
color2.jpg&nbsp;&nbsp;&nbsp;&nbsp;*the color image of the subordinate camera* <br /><br />
depth1.png&nbsp;&nbsp;&nbsp;&nbsp;*the depth image of the master camera* <br /><br />
depth2.png&nbsp;&nbsp;&nbsp;*the depth image of the subordinate camera*

The script using externally acquired extrinsic matrices requires the following on top of the input mentioned above:
extrinsic.json&nbsp;&nbsp;&nbsp;&nbsp;*the camera extrinsic matrices, for the transformation fom sub camera to master camera* <br /><br />
