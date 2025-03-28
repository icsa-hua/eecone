from pathlib import Path
import numpy as np
import cv2
from datetime import datetime
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

bagpath = Path('C:/Users/k.chiotis/Downloads/rosbag2_2023_09_25-17_15_00')
# C:/Users/k.chiotis/Downloads/rosbag2_2023_09_25-17_15_00
# C:/Users/k.chiotis/Downloads/rosbag2_burning_car_1

# Create a type store to use if the bag has no message definitions.
typestore = get_typestore(Stores.ROS2_FOXY)

topics = ['/radar_visualization_markers']

# Create reader instance and open for reading.
with AnyReader([bagpath], default_typestore=typestore) as reader:
    connections = [x for x in reader.connections if x.topic in topics]
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        # Extract image data
        frame_id = msg.header.frame_id
        dt = datetime.fromtimestamp(timestamp / 1e9).strftime("%Y_%m_%d_%H_%M_%S")
        print(f"Received image from {frame_id} at {dt}")
        height = msg.height
        width = msg.width
        encoding = msg.encoding
        if encoding == 'rgb8':
            data = np.array(msg.data, dtype=np.uint8).reshape((height, width, 3))
        elif encoding == 'mono8':
            data = np.array(msg.data, dtype=np.uint8).reshape((height, width))
        else:
            print(f"Unsupported encoding: {encoding}")
            continue
        
        # save the image as a file
        # image_path = f"C:/Users/k.chiotis/OneDrive - Titan Cement Company SA/Desktop/rosbag images/big file/thermal_image_color_max_range/{dt}_{frame_id}_image.png"
        # cv2.imwrite(image_path, data)

        # Display the image
        # cv2.imshow(f"Image from {frame_id} at {dt}", data)
        # cv2.waitKey(0)  # Wait for a key press to close the window
        # cv2.destroyAllWindows()