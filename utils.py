import cv2
import numpy as np

from pyorbbecsdk import OBFormat, Frame

def frame_to_bgr_image(frame: Frame):
    """
    Converts a frame to a BGR image.
    """
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    frame_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if frame_format == OBFormat.RGB:
        data = data.reshape((height, width, 3))
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        return image
    if frame_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    if frame_format == OBFormat.YUYV:
        data = data.reshape(height, width, 2)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_YUYV)
        return image
    if frame_format == OBFormat.NV21:
        data = data.reshape((height * 3 // 2, width))
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_NV21)
        return image
    if frame_format == OBFormat.I420:
        data = data.reshape((height * 3 // 2, width))
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_I420)
        return image
    if frame_format == OBFormat.Y16:
        data = data.reshape((height, width, 2))
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_YUYV)
        return image
    if frame_format == OBFormat.GRAY8:
        data = data.reshape((height, width, 1))
        image = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        return image
    return None
