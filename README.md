# Mantis 2.0
This repository contains utility functions for processing and visualizing multi-camera and multi-sensor tracking data. Specifically, it includes funcitonality for using the Orbbec Astra's Depth, IR, and RGBD feed and synchronizing those output streams with another USB camera that outputs RGBD.


## Dependencies

Install the following python packages before using the utilities 

```bash
pip install numpy matplotlib opencv-python opencv-python pillor pyorbbecsdk
```
Also important to clone these python bindings provided by Orbbec and use the main branch for version 1 if using an Orbbec Astra https://github.com/orbbec/pyorbbecsdk
![image](https://github.com/user-attachments/assets/f4d83bfd-2d4a-4e25-a858-4bb8f720b8b2)
Videos are synchronized and can be verified via system time on the top left corner of each threaded feed.
Start and stop buttons record 4 videos simultaneously and save them into 4 files.
