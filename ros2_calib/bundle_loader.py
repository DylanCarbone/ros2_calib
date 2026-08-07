# MIT License
#
# Copyright (c) 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Loader for LiDAR-to-Camera calibration inputs, consisting of exactly three
files:
  - a point cloud (.laz/.las)
  - a camera image (.jpg/.png/...)
  - a calibration metadata bundle (.npz) as saved by lidarcam's
    bundle_metadata(), containing 'intrinsics_matrix', 'distortion_matrix'
    and 'extrinsics_matrix'.

The loaded data is wrapped in the same mock ROS message dataclasses used
elsewhere in this app (see ros_utils.py) so it can flow through the existing
LiDAR2Cam calibration widget unchanged.
"""

import os

import cv2
import laspy
import numpy as np

from . import ros_utils

IMAGE_TOPIC = "/scene/image"
POINTCLOUD_TOPIC = "/scene/points"
CAMERAINFO_TOPIC = "/scene/camera_info"

LIDAR_FRAME_ID = "lidar"
CAMERA_FRAME_ID = "camera"

REQUIRED_BUNDLE_KEYS = ("intrinsics_matrix", "distortion_matrix", "extrinsics_matrix")


def _load_pointcloud(path: str) -> ros_utils.PointCloud2:
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    intensity = np.asarray(getattr(las, "intensity", np.zeros_like(x)), dtype=np.float32)

    num_points = x.shape[0]
    structured = np.zeros(
        num_points,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")],
    )
    structured["x"] = x
    structured["y"] = y
    structured["z"] = z
    structured["intensity"] = intensity

    fields = [
        ros_utils.PointField(name="x", offset=0, datatype=ros_utils.PointField.FLOAT32, count=1),
        ros_utils.PointField(name="y", offset=4, datatype=ros_utils.PointField.FLOAT32, count=1),
        ros_utils.PointField(name="z", offset=8, datatype=ros_utils.PointField.FLOAT32, count=1),
        ros_utils.PointField(
            name="intensity", offset=12, datatype=ros_utils.PointField.FLOAT32, count=1
        ),
    ]

    return ros_utils.PointCloud2(
        header=ros_utils.Header(frame_id=LIDAR_FRAME_ID),
        height=1,
        width=num_points,
        fields=fields,
        is_bigendian=False,
        point_step=16,
        row_step=16 * num_points,
        data=structured.tobytes(),
        is_dense=True,
    )


def _load_image(path: str) -> ros_utils.Image:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image file: {path}")
    height, width = image_bgr.shape[:2]

    return ros_utils.Image(
        header=ros_utils.Header(frame_id=CAMERA_FRAME_ID),
        height=height,
        width=width,
        encoding="bgr8",
        is_bigendian=False,
        step=width * 3,
        data=np.ascontiguousarray(image_bgr).tobytes(),
    )


def find_missing_bundle_keys(bundle_path: str) -> list:
    """Return a list of required npz keys that are missing from bundle_path."""
    with np.load(bundle_path) as npz:
        return [key for key in REQUIRED_BUNDLE_KEYS if key not in npz]


def _load_camera_info(
    bundle_path: str, fallback_width: int, fallback_height: int
) -> tuple:
    """Returns (CameraInfo, extrinsics_matrix, intrinsics_matrix, distortion_matrix)."""
    missing = find_missing_bundle_keys(bundle_path)
    if missing:
        raise ValueError(
            f"'{os.path.basename(bundle_path)}' is missing required array(s): "
            f"{', '.join(missing)}"
        )

    with np.load(bundle_path) as npz:
        intrinsics_matrix = np.asarray(npz["intrinsics_matrix"], dtype=np.float64).reshape(3, 3)
        distortion_matrix = np.asarray(npz["distortion_matrix"], dtype=np.float64).flatten()
        extrinsics_matrix = np.asarray(npz["extrinsics_matrix"], dtype=np.float64).reshape(4, 4)
        width = int(npz["width"]) if "width" in npz else fallback_width
        height = int(npz["height"]) if "height" in npz else fallback_height

    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:3, :3] = intrinsics_matrix

    distortion_model = {5: "plumb_bob", 8: "rational_polynomial"}.get(
        len(distortion_matrix), "plumb_bob"
    )

    camerainfo_msg = ros_utils.CameraInfo(
        header=ros_utils.Header(frame_id=CAMERA_FRAME_ID),
        height=height,
        width=width,
        distortion_model=distortion_model,
        d=distortion_matrix.tolist(),
        k=intrinsics_matrix.flatten().tolist(),
        r=np.eye(3, dtype=np.float64).flatten().tolist(),
        p=projection.flatten().tolist(),
    )

    return camerainfo_msg, extrinsics_matrix, intrinsics_matrix, distortion_matrix


def load_calibration_bundle(laz_path: str, image_path: str, bundle_path: str) -> dict:
    """
    Load a LiDAR2Cam calibration input set (point cloud + image + metadata
    bundle) into the same data structures produced by the rosbag-based
    loading path, so it can be handed directly to the transform-selection
    and calibration widgets.

    Also returns the raw extrinsics/intrinsics/distortion matrices from the
    bundle so the caller can pre-fill the initial transform and later
    re-save a finalised bundle.
    """
    print(f"Using point cloud from: {laz_path}")
    pointcloud_msg = _load_pointcloud(laz_path)

    print(f"Using image from: {image_path}")
    image_msg = _load_image(image_path)

    print(f"Using intrinsics_matrix, distortion_matrix and extrinsics_matrix from: {bundle_path}")
    camerainfo_msg, extrinsics_matrix, intrinsics_matrix, distortion_matrix = _load_camera_info(
        bundle_path,
        fallback_width=image_msg.width,
        fallback_height=image_msg.height,
    )

    topic_types = {
        IMAGE_TOPIC: "sensor_msgs/msg/Image",
        POINTCLOUD_TOPIC: "sensor_msgs/msg/PointCloud2",
        CAMERAINFO_TOPIC: "sensor_msgs/msg/CameraInfo",
    }

    return {
        "image_topic": IMAGE_TOPIC,
        "pointcloud_topic": POINTCLOUD_TOPIC,
        "camerainfo_topic": CAMERAINFO_TOPIC,
        "topic_types": topic_types,
        "raw_messages": {
            IMAGE_TOPIC: image_msg,
            POINTCLOUD_TOPIC: pointcloud_msg,
            CAMERAINFO_TOPIC: camerainfo_msg,
        },
        "lidar_frame": LIDAR_FRAME_ID,
        "camera_frame": CAMERA_FRAME_ID,
        "image_msg": image_msg,
        "pointcloud_msg": pointcloud_msg,
        "camerainfo_msg": camerainfo_msg,
        "extrinsics_matrix": extrinsics_matrix,
        "intrinsics_matrix": intrinsics_matrix,
        "distortion_matrix": distortion_matrix,
    }
