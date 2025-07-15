# Multi-Camera Stream Portable

This repository contains a portable version of the multi-camera and multi-sensor tracking data processing and visualization utilities. It includes functionality for using the Orbbec Astra's Depth, IR, and RGBD feed and synchronizing those output streams with another USB camera that outputs RGBD.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd multi-cam-stream-portable
    ```

2.  **Install Python Dependencies:**
    It is highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Install Orbbec Python Bindings:**
    This project relies on the `pyorbbecsdk` library. You need to clone their Python bindings and use the main branch for version 1 if you are using an Orbbec Astra camera.

    ```bash
    git clone https://github.com/orbbec/pyorbbecsdk.git
    cd pyorbbecsdk
    # Follow their installation instructions, typically:
    pip install .
    # Or refer to their documentation for specific build steps if needed.
    cd ..
    ```

4.  **Ensure `v4l2-ctl` is installed (for Logitech camera detection):**
    This utility is used to detect the Logitech camera index.
    ```bash
    sudo apt-get install v4l-utils
    ```

## Running the Application

To start the multi-camera recorder, activate your virtual environment and run the main script:

```bash
source venv/bin/activate
python multi_cam_recorder_cross_platform.py
```

## Configuration

The `multi_device_sync_config.json` file contains configuration settings, including the `recordings_directory`. By default, recordings will be saved in a `recordings` folder within the project directory. You can modify this path within the `multi_device_sync_config.json` file if needed.

```json
{
    "recordings_directory": "recordings",
    "devices": [
        {
            "serial_number": "CL838420074",
            "config": {
                "mode": "FREE_RUN",
                "depth_delay_us": 0,
                "color_delay_us": 0,
                "trigger_out_enable": false,
                "trigger_out_delay_us": 0,
                "frames_per_trigger": 1
            }
        },
        {
            "serial_number": "CL8M84100SX",
            "config": {
                "mode": "FREE_RUN",
                "depth_delay_us": 0,
                "color_delay_us": 0,
                "trigger_out_enable": false,
                "trigger_out_delay_us": 0,
                "frames_per_trigger": 1
            }
        }
    ]
}
```