import json
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime
from pyorbbecsdk import *
import numpy as np
from functools import partial

# Load configuration
with open('/home/kaliber/multi-cam-stream/multi_device_sync_config.json', 'r') as f:
    config = json.load(f)

recordings_dir = config.get('recordings_directory', '/home/kaliber/multi-cam-stream/recordings')

# Global variables
is_recording = False
logitech_out = None
orbbec_rgb_out = None
orbbec_ir_out = None
orbbec_depth_out = None
stop_event = threading.Event()
logitech_cam = None
orbbec_pipeline = None

# --- Utility Functions (adapted from pyorbbecsdk examples) ---
def frame_to_bgr_image(frame):
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
    elif frame_format == OBFormat.Y16:
        data.dtype = np.uint16
        data = data.reshape((height, width))
        # Scale data to 8-bit for display
        data = cv2.convertScaleAbs(data, alpha=0.05)
        image = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        return image
    elif frame_format == OBFormat.GRAY:
        data = data.reshape((height, width, 1))
        image = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        return image
    elif frame_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    else:
        # Fallback for other formats, might need more specific handling
        print(f"Unsupported frame format in frame_to_bgr_image: {frame_format}")
        return None

def process_depth(frame):
    if not frame:
        return None
    try:
        depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape(frame.get_height(), frame.get_width())
        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    except ValueError:
        return None

def process_ir(ir_frame):
    if ir_frame is None:
        return None
    ir_frame = ir_frame.as_video_frame()
    ir_data = np.asanyarray(ir_frame.get_data())
    width = ir_frame.get_width()
    height = ir_frame.get_height()
    ir_format = ir_frame.get_format()

    if ir_format == OBFormat.Y8:
        ir_data = np.resize(ir_data, (height, width, 1))
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
    elif ir_format == OBFormat.MJPG:
        ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
        if ir_data is None:
            print("decode mjpeg failed")
            return None
        ir_data = np.resize(ir_data, (height, width, 1))
    elif ir_format == OBFormat.GRAY:
        ir_data = np.resize(ir_data, (height, width, 1))
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
    else:
        ir_data = np.frombuffer(ir_data, dtype=np.uint16)
        data_type = np.uint16
        image_dtype = cv2.CV_16UC1
        max_data = 255
        ir_data = np.resize(ir_data, (height, width, 1))

    cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
    ir_data = ir_data.astype(data_type)
    return cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)

# --- GUI Setup ---
root = tk.Tk()
root.title("Multi-camera Recorder")
root.geometry("1920x1080") # Increased size for multiple previews

main_frame = ttk.Frame(root)
main_frame.grid(row=0, column=0, sticky="nsew")

# Logitech Frame
logitech_frame = ttk.LabelFrame(main_frame, text="Logitech Camera (RGB)")
logitech_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
logitech_label = ttk.Label(logitech_frame)
logitech_label.pack()

# Orbbec RGB Frame
orbbec_rgb_frame = ttk.LabelFrame(main_frame, text="Orbbec Femto (RGB)")
orbbec_rgb_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
orbbec_rgb_label = ttk.Label(orbbec_rgb_frame)
orbbec_rgb_label.pack()

# Orbbec IR Frame
orbbec_ir_frame = ttk.LabelFrame(main_frame, text="Orbbec Femto (IR)")
orbbec_ir_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
orbbec_ir_label = ttk.Label(orbbec_ir_frame)
orbbec_ir_label.pack()

# Orbbec Depth Frame
orbbec_depth_frame = ttk.LabelFrame(main_frame, text="Orbbec Femto (Depth)")
orbbec_depth_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
orbbec_depth_label = ttk.Label(orbbec_depth_frame)
orbbec_depth_label.pack()

# Configure grid weights to make frames expand
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_rowconfigure(1, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

button_frame = ttk.Frame(root)
button_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# --- Functions ---

def update_label(label, imgtk):
    """Thread-safe way to update a Tkinter label with a new image."""
    label.imgtk = imgtk
    label.configure(image=imgtk)

def start_recording():
    global is_recording, logitech_out, orbbec_rgb_out, orbbec_ir_out, orbbec_depth_out
    if not is_recording:
        # Create recordings directory if it doesn't exist
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)

        # Setup Logitech recorder
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        logitech_out = cv2.VideoWriter(os.path.join(recordings_dir, 'logitech_output.avi'), fourcc, 30.0, (640, 480))

        # Setup Orbbec recorders
        orbbec_rgb_out = cv2.VideoWriter(os.path.join(recordings_dir, 'orbbec_rgb_output.avi'), fourcc, 30.0, (640, 480))
        orbbec_ir_out = cv2.VideoWriter(os.path.join(recordings_dir, 'orbbec_ir_output.avi'), fourcc, 30.0, (640, 480), isColor=False) # IR is grayscale
        orbbec_depth_out = cv2.VideoWriter(os.path.join(recordings_dir, 'orbbec_depth_output.avi'), fourcc, 30.0, (640, 480), isColor=False) # Depth is grayscale
        
        is_recording = True
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        print("Recording started.")

def stop_recording():
    global is_recording, logitech_out, orbbec_rgb_out, orbbec_ir_out, orbbec_depth_out
    if is_recording:
        is_recording = False
        if logitech_out:
            logitech_out.release()
            logitech_out = None
        if orbbec_rgb_out:
            orbbec_rgb_out.release()
            orbbec_rgb_out = None
        if orbbec_ir_out:
            orbbec_ir_out.release()
            orbbec_ir_out = None
        if orbbec_depth_out:
            orbbec_depth_out.release()
            orbbec_depth_out = None
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        print("Recording stopped.")

def logitech_camera_thread():
    global logitech_cam, logitech_out, is_recording
    found_camera = False
    # Try specific index /dev/video6 first, then iterate
    indices_to_try = [6] + [i for i in range(10) if i != 6] # Try 6 first, then 0-5, 7-9
    for i in indices_to_try:
        print(f"Attempting to open Logitech camera at index {i}...")
        logitech_cam = cv2.VideoCapture(i)
        if logitech_cam.isOpened():
            print(f"Logitech camera opened successfully at index {i}.")
            found_camera = True
            break
        else:
            logitech_cam.release()
    
    if not found_camera:
        print("Failed to open Logitech camera after trying multiple indices.")
        return

    while not stop_event.is_set():
        ret, frame = logitech_cam.read()
        if ret:
            # Resize the frame once for both preview and recording
            resized_frame = cv2.resize(frame, (640, 480))

            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(resized_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Preview
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            # Schedule GUI update on the main thread
            root.after(1, partial(update_label, logitech_label, imgtk))

            # Recording
            if is_recording and logitech_out:
                logitech_out.write(resized_frame)
    
    logitech_cam.release()

def orbbec_camera_thread():
    global orbbec_pipeline, orbbec_rgb_out, orbbec_ir_out, orbbec_depth_out, is_recording
    try:
        ctx = Context()
        device_list = ctx.query_devices()
        if device_list.get_count() == 0:
            print("No Orbbec device connected.")
            return
        
        device = device_list.get_device_by_index(0)
        orbbec_pipeline = Pipeline(device)
        config = Config()

        # Enumerate and configure all video sensors
        sensor_list = device.get_sensor_list()
        for i in range(sensor_list.get_count()):
            sensor = sensor_list.get_sensor_by_index(i)
            sensor_type = sensor.get_type()
            
            # Only enable video streams (Color, IR, Depth)
            if sensor_type in [OBSensorType.COLOR_SENSOR, OBSensorType.IR_SENSOR, OBSensorType.DEPTH_SENSOR]:
                try:
                    config.enable_stream(sensor_type)
                    print(f"Enabled {sensor_type.name} stream.")
                except OBError as e:
                    print(f"Failed to enable {sensor_type.name} stream: {e}")
                    # Continue to try other streams even if one fails

        orbbec_pipeline.start(config)
        
        while not stop_event.is_set():
            frames = orbbec_pipeline.wait_for_frames(100)
            if frames:
                # RGB Frame (for preview and recording)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_bgr = frame_to_bgr_image(color_frame)
                    if frame_bgr is not None:
                        resized_frame_bgr = cv2.resize(frame_bgr, (640, 480))
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        cv2.putText(resized_frame_bgr, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        frame_rgb = cv2.cvtColor(resized_frame_bgr, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        imgtk = ImageTk.PhotoImage(image=img)
                        root.after(1, partial(update_label, orbbec_rgb_label, imgtk))
                        if is_recording and orbbec_rgb_out:
                            orbbec_rgb_out.write(resized_frame_bgr)

                # IR Frame (for preview and recording)
                ir_frame = frames.get_ir_frame()
                if ir_frame:
                    ir_image = process_ir(ir_frame)
                    if ir_image is not None:
                        resized_ir_image = cv2.resize(ir_image, (640, 480))
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        cv2.putText(resized_ir_image, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        img = Image.fromarray(resized_ir_image)
                        imgtk = ImageTk.PhotoImage(image=img)
                        root.after(1, partial(update_label, orbbec_ir_label, imgtk))
                        if is_recording and orbbec_ir_out:
                            # VideoWriter for grayscale expects a 2D array
                            if len(resized_ir_image.shape) == 3 and resized_ir_image.shape[2] == 3:
                                gray_frame = cv2.cvtColor(resized_ir_image, cv2.COLOR_BGR2GRAY)
                                orbbec_ir_out.write(gray_frame)
                            else:
                                orbbec_ir_out.write(resized_ir_image)

                # Depth Frame (for preview and recording)
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = process_depth(depth_frame)
                    if depth_image is not None:
                        resized_depth_image = cv2.resize(depth_image, (640, 480))
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        cv2.putText(resized_depth_image, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        img = Image.fromarray(resized_depth_image)
                        imgtk = ImageTk.PhotoImage(image=img)
                        root.after(1, partial(update_label, orbbec_depth_label, imgtk))
                        if is_recording and orbbec_depth_out:
                            # VideoWriter for grayscale expects a 2D array
                            if len(resized_depth_image.shape) == 3 and resized_depth_image.shape[2] == 3:
                                gray_frame = cv2.cvtColor(resized_depth_image, cv2.COLOR_BGR2GRAY)
                                orbbec_depth_out.write(gray_frame)
                            else:
                                orbbec_depth_out.write(resized_depth_image)

    except OBError as e:
        print(f"Orbbec pipeline error: {e}")
    finally:
        if orbbec_pipeline:
            orbbec_pipeline.stop()

def on_closing():
    print("Closing application...")
    stop_event.set()
    # Wait a moment for threads to see the stop event
    root.after(100, root.destroy)

# --- Button Bindings ---
start_button = ttk.Button(button_frame, text="Start Recording", command=start_recording)
start_button.pack(side=tk.LEFT, padx=20)

stop_button = ttk.Button(button_frame, text="Stop Recording", command=stop_recording, state=tk.DISABLED)
stop_button.pack(side=tk.RIGHT, padx=20)

# --- Main Loop ---
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start camera threads for preview
threading.Thread(target=logitech_camera_thread, daemon=True).start()
threading.Thread(target=orbbec_camera_thread, daemon=True).start()

root.mainloop()

# Cleanly release resources on exit (optional, as daemon threads will exit)
if logitech_cam and logitech_cam.isOpened():
    logitech_cam.release()
if orbbec_pipeline:
    orbbec_pipeline.stop()
