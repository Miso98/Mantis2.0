
import cv2
import numpy as np
from pyorbbecsdk import *

from utils import frame_to_bgr_image

ESC_KEY = 27

def main():
    try:
        ctx = Context()
        device_list = ctx.query_devices()
        if device_list.get_count() == 0:
            print("No device connected.")
            return

        print("Attempting to open the first available device...")
        device = device_list.get_device_by_index(0)
        pipeline = Pipeline(device)
        config = Config()

        # Try to get the color stream profile
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        
        pipeline.start(config)
        print("Pipeline started successfully!")

        while True:
            frames = pipeline.wait_for_frames(100)
            if frames:
                color_frame = frames.get_color_frame()
                if color_frame:
                    img = frame_to_bgr_image(color_frame)
                    if img is not None:
                        cv2.imshow("Single Camera Test", img)
            
            key = cv2.waitKey(1)
            if key == ESC_KEY:
                break

    except OBError as e:
        print(f"An OBError occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Stopping pipeline...")
        if 'pipeline' in locals() and pipeline:
            try: pipeline.stop() 
            except: pass
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
