# Mantis 2.0
This repository contains utility functions for processing and visualizing multi-camera and multi-sensor tracking data. Specifically, it includes funcitonality for using the Orbbec Astra's Depth, IR, and RGBD feed and synchronizing those output streams with another USB camera that outputs RGBD.


## Dependencies

Install the following python packages before using the utilities 

'''bash
pip install numpy matplotlib opencv-python

Also important to clone these python bindings provided by Orbbec and use the main branch for version 1 if using an Orbbec Astra https://github.com/orbbec/pyorbbecsdk
