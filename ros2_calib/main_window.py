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

import os

import numpy as np
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import ros_utils
from . import tf_transformations as tf
from .bundle_loader import (
    is_image_points_csv,
    load_calibration_bundle,
    load_marked_image_points,
    load_marked_points,
)
from .calibration_widget import CalibrationWidget
from .common import UIStyles

POINTCLOUD_EXTENSIONS = (".laz", ".las")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
BUNDLE_EXTENSIONS = (".npz",)
POINTS_EXTENSIONS = (".csv",)


class MainWindow(QMainWindow):
    def __init__(
        self,
        laz_path=None,
        image_path=None,
        bundle_path=None,
        points_path=None,
        image_points_path=None,
    ):
        super().__init__()
        self.setWindowTitle("ros2_calib - LiDAR-Camera Calibration Tool")
        self.setGeometry(100, 100, 1800, 800)
        self.setAcceptDrops(True)

        # Loaded input paths
        self.laz_path = None
        self.image_path = None
        self.bundle_path = None
        # Optional .csv of the LiDAR points the initial extrinsics were solved from
        self.points_path = None
        # Optional .csv of the matching image points, in the same order, each pair of which is
        # added as a correspondence
        self.image_points_path = None

        # Data produced from the loaded inputs
        self.image_msg = None
        self.pointcloud_msg = None
        self.camerainfo_msg = None
        self.intrinsics_matrix = None
        self.distortion_matrix = None
        self.marked_points = None
        self.marked_image_points = None

        # Current transform is kept in the "camera pose expressed in the
        # lidar frame" convention expected by CalibrationWidget's
        # initial_transform argument (it is the inverse of the
        # lidar->camera extrinsics_matrix convention used by the bundle).
        self.current_transform = np.eye(4, dtype=np.float64)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.setup_load_view()
        self.setup_transform_view()

        self.stacked_widget.setCurrentIndex(0)

        if laz_path:
            self.set_laz_path(laz_path)
        if image_path:
            self.set_image_path(image_path)
        if bundle_path:
            self.set_bundle_path(bundle_path)
        if points_path:
            self.set_points_path(points_path)
        if image_points_path:
            self.set_image_points_path(image_points_path)

    # ------------------------------------------------------------------
    # Load view
    # ------------------------------------------------------------------
    def setup_load_view(self):
        self.load_widget = QWidget()
        self.load_layout = QVBoxLayout(self.load_widget)

        title_label = QLabel("Load LiDAR-Camera Calibration Data")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        self.load_layout.addWidget(title_label)
        desc_label = QLabel(
            "Load a point cloud (.laz/.las), a camera image, and the calibration "
            "metadata bundle (.npz) saved by bundle_metadata(). All three are required. "
            "Optionally, also load the LiDAR points (.csv) the initial extrinsics were "
            "solved from, to highlight them in blue, and their matching image points (.csv), "
            "to start with each pair selected as a correspondence."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        self.load_layout.addWidget(desc_label)

        inputs_group = QGroupBox("Required Inputs")
        inputs_layout = QGridLayout(inputs_group)

        self.load_laz_button = QPushButton("Load Point Cloud (.laz)")
        self.load_laz_button.clicked.connect(self.load_laz_dialog)
        self.laz_path_label = QLabel("No point cloud loaded.")
        self.laz_path_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        inputs_layout.addWidget(self.load_laz_button, 0, 0)
        inputs_layout.addWidget(self.laz_path_label, 0, 1)

        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image_dialog)
        self.image_path_label = QLabel("No image loaded.")
        self.image_path_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        inputs_layout.addWidget(self.load_image_button, 1, 0)
        inputs_layout.addWidget(self.image_path_label, 1, 1)

        self.load_bundle_button = QPushButton("Load Calibration Bundle (.npz)")
        self.load_bundle_button.clicked.connect(self.load_bundle_dialog)
        self.bundle_path_label = QLabel("No calibration bundle loaded.")
        self.bundle_path_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        inputs_layout.addWidget(self.load_bundle_button, 2, 0)
        inputs_layout.addWidget(self.bundle_path_label, 2, 1)

        self.load_points_button = QPushButton("Load Initial Extrinsics LiDAR Points (.csv)")
        self.load_points_button.clicked.connect(self.load_points_dialog)
        self.points_path_label = QLabel("Optional: no LiDAR points loaded.")
        self.points_path_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        inputs_layout.addWidget(self.load_points_button, 3, 0)
        inputs_layout.addWidget(self.points_path_label, 3, 1)

        self.load_image_points_button = QPushButton("Load Initial Extrinsics Image Points (.csv)")
        self.load_image_points_button.clicked.connect(self.load_image_points_dialog)
        self.image_points_path_label = QLabel("Optional: no image points loaded.")
        self.image_points_path_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        inputs_layout.addWidget(self.load_image_points_button, 4, 0)
        inputs_layout.addWidget(self.image_points_path_label, 4, 1)

        inputs_layout.setColumnStretch(1, 1)
        self.load_layout.addWidget(inputs_group)

        drag_drop_label = QLabel("or Drag & Drop the files anywhere in this window")
        drag_drop_label.setStyleSheet("color: #666; font-style: italic;")
        self.load_layout.addWidget(drag_drop_label)

        self.load_status_label = QLabel("")
        self.load_status_label.setStyleSheet("padding: 5px; color: #b00020;")
        self.load_layout.addWidget(self.load_status_label)

        self.load_layout.addStretch()

        proceed_layout = QHBoxLayout()
        proceed_layout.addStretch()
        self.proceed_button = QPushButton("Proceed to Transform Selection")
        self.proceed_button.setEnabled(False)
        self.proceed_button.clicked.connect(self.proceed_to_transform_view)
        self.proceed_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        proceed_layout.addWidget(self.proceed_button)
        self.load_layout.addLayout(proceed_layout)

        self.stacked_widget.addWidget(self.load_widget)

    def load_laz_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud", "", "Point Cloud Files (*.laz *.las)"
        )
        if file_path:
            self.set_laz_path(file_path)

    def load_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.set_image_path(file_path)

    def load_bundle_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Calibration Bundle", "", "NumPy Bundle (*.npz)"
        )
        if file_path:
            self.set_bundle_path(file_path)

    def load_points_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Initial Extrinsics LiDAR Points", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.set_points_path(file_path)

    def load_image_points_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Initial Extrinsics Image Points", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.set_image_points_path(file_path)

    def set_laz_path(self, path):
        self.laz_path = path
        self.laz_path_label.setText(path)
        self.update_proceed_button_state()

    def set_image_path(self, path):
        self.image_path = path
        self.image_path_label.setText(path)
        self.update_proceed_button_state()

    def set_bundle_path(self, path):
        self.bundle_path = path
        self.bundle_path_label.setText(path)
        self.update_proceed_button_state()

    def set_points_path(self, path):
        self.points_path = path
        self.points_path_label.setText(path)

    def set_image_points_path(self, path):
        self.image_points_path = path
        self.image_points_path_label.setText(path)

    def update_proceed_button_state(self):
        self.proceed_button.setEnabled(
            bool(self.laz_path and self.image_path and self.bundle_path)
        )

    def proceed_to_transform_view(self):
        self.load_status_label.setText("")
        try:
            data = load_calibration_bundle(self.laz_path, self.image_path, self.bundle_path)
        except Exception as e:
            self.load_status_label.setText(f"Error loading calibration data: {e}")
            return

        self.marked_points = None
        if self.points_path:
            try:
                self.marked_points = load_marked_points(self.points_path)
            except Exception as e:
                self.load_status_label.setText(f"Error loading initial extrinsics points: {e}")
                return
            print(f"Highlighting {len(self.marked_points)} LiDAR points from: {self.points_path}")

        self.marked_image_points = None
        if self.image_points_path:
            if self.marked_points is None:
                self.load_status_label.setText(
                    "Error: the initial extrinsics image points need their matching LiDAR points"
                )
                return
            try:
                self.marked_image_points = load_marked_image_points(self.image_points_path)
            except Exception as e:
                self.load_status_label.setText(
                    f"Error loading initial extrinsics image points: {e}"
                )
                return
            if len(self.marked_image_points) != len(self.marked_points):
                self.load_status_label.setText(
                    f"Error: {len(self.marked_image_points)} initial extrinsics image points but "
                    f"{len(self.marked_points)} LiDAR points; they pair up row by row"
                )
                return
            print(
                f"Adding {len(self.marked_image_points)} correspondences from: "
                f"{self.image_points_path}"
            )

        self.image_msg = data["image_msg"]
        self.pointcloud_msg = data["pointcloud_msg"]
        self.camerainfo_msg = data["camerainfo_msg"]
        self.intrinsics_matrix = data["intrinsics_matrix"]
        self.distortion_matrix = data["distortion_matrix"]

        # data["extrinsics_matrix"] is in the lidar->camera convention
        # (p_cam = R @ p_lidar + t); CalibrationWidget's initial_transform
        # expects the inverse of that.
        self.current_transform = np.linalg.inv(data["extrinsics_matrix"])
        self.update_transform_display()
        self.update_manual_inputs_from_matrix()
        self.stacked_widget.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Transform view
    # ------------------------------------------------------------------
    def setup_transform_view(self):
        self.transform_widget = QWidget()
        self.transform_layout = QVBoxLayout(self.transform_widget)

        back_button_layout = QHBoxLayout()
        self.back_to_load_button = QPushButton("← Back to Load")
        self.back_to_load_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        back_button_layout.addWidget(self.back_to_load_button)
        back_button_layout.addStretch()
        self.transform_layout.addLayout(back_button_layout)

        manual_group = QGroupBox("Initial Transform (from calibration bundle)")
        manual_layout = QGridLayout(manual_group)
        self.tx_input, self.ty_input, self.tz_input = (
            QLineEdit("0.0"),
            QLineEdit("0.0"),
            QLineEdit("0.0"),
        )
        self.rx_input, self.ry_input, self.rz_input = (
            QLineEdit("0.0"),
            QLineEdit("0.0"),
            QLineEdit("0.0"),
        )
        manual_layout.addWidget(QLabel("Translation (x, y, z):"), 0, 0)
        manual_layout.addWidget(self.tx_input, 0, 1)
        manual_layout.addWidget(self.ty_input, 0, 2)
        manual_layout.addWidget(self.tz_input, 0, 3)
        manual_layout.addWidget(QLabel("Rotation (roll, pitch, yaw) [rad]:"), 1, 0)
        manual_layout.addWidget(self.rx_input, 1, 1)
        manual_layout.addWidget(self.ry_input, 1, 2)
        manual_layout.addWidget(self.rz_input, 1, 3)
        self.update_manual_button = QPushButton("Update Transform from Manual Input")
        self.update_manual_button.clicked.connect(self.update_manual_transform)
        manual_layout.addWidget(self.update_manual_button, 2, 0, 1, 4)
        self.transform_layout.addWidget(manual_group)

        transform_group = QGroupBox("Current Transformation Matrix")
        transform_layout_inner = QVBoxLayout(transform_group)
        self.transform_display = QTextEdit(maximumHeight=180, readOnly=True, fontFamily="monospace")
        transform_layout_inner.addWidget(self.transform_display)
        self.transform_layout.addWidget(transform_group)

        button_layout = QHBoxLayout()
        self.use_identity_button = QPushButton("Use Identity Transform")
        self.use_identity_button.clicked.connect(self.use_identity_transform)
        button_layout.addWidget(self.use_identity_button)
        button_layout.addStretch()
        self.confirm_button = QPushButton("Start Calibration")
        self.confirm_button.clicked.connect(self.confirm_transformation)
        self.confirm_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        button_layout.addWidget(self.confirm_button)
        self.transform_layout.addLayout(button_layout)
        self.update_transform_display()
        self.stacked_widget.addWidget(self.transform_widget)

    def update_manual_transform(self):
        try:
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_euler(
                "xyz",
                [
                    float(self.rx_input.text()),
                    float(self.ry_input.text()),
                    float(self.rz_input.text()),
                ],
            )
            self.current_transform = np.eye(4, dtype=np.float64)
            self.current_transform[:3, :3] = rot.as_matrix()
            self.current_transform[:3, 3] = [
                float(self.tx_input.text()),
                float(self.ty_input.text()),
                float(self.tz_input.text()),
            ]
            self.update_transform_display()
        except ValueError as e:
            print(f"[ERROR] Invalid manual transform input: {e}")

    def update_manual_inputs_from_matrix(self):
        trans = tf.translation_from_matrix(self.current_transform)
        euler = tf.euler_from_matrix(self.current_transform)
        self.tx_input.setText(f"{trans[0]:.6f}")
        self.ty_input.setText(f"{trans[1]:.6f}")
        self.tz_input.setText(f"{trans[2]:.6f}")
        self.rx_input.setText(f"{euler[0]:.6f}")
        self.ry_input.setText(f"{euler[1]:.6f}")
        self.rz_input.setText(f"{euler[2]:.6f}")

    def use_identity_transform(self):
        self.current_transform = np.eye(4, dtype=np.float64)
        self.update_transform_display()
        self.update_manual_inputs_from_matrix()

    def update_transform_display(self):
        matrix_str = "\n".join(
            ["  ".join([f"{val:8.4f}" for val in row]) for row in self.current_transform]
        )
        trans = tf.translation_from_matrix(self.current_transform)
        rpy = tf.euler_from_matrix(self.current_transform)

        combined_display = f"Transformation Matrix:\n{matrix_str}\n\nTranslation and Rotation:\nXYZ: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]\nRPY: [{rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}]"
        self.transform_display.setPlainText(combined_display)

    def confirm_transformation(self):
        self.proceed_to_calibration(self.current_transform)

    # ------------------------------------------------------------------
    # Calibration widget
    # ------------------------------------------------------------------
    def proceed_to_calibration(self, initial_transform):
        for i in range(self.stacked_widget.count() - 1, 1, -1):
            widget = self.stacked_widget.widget(i)
            self.stacked_widget.removeWidget(widget)
            widget.deleteLater()

        self.calibration_widget = CalibrationWidget(
            self.image_msg,
            self.pointcloud_msg,
            self.camerainfo_msg,
            ros_utils,
            initial_transform,
            marked_points_xyz=self.marked_points,
            marked_image_points=self.marked_image_points,
        )
        self.calibration_widget.calibration_completed.connect(self.on_calibration_completed)
        self.stacked_widget.setCurrentIndex(self.stacked_widget.addWidget(self.calibration_widget))

    def on_calibration_completed(self, calibration_results):
        """Called when the user clicks 'Export Calibration' in the calibration widget."""
        if isinstance(calibration_results, dict):
            extrinsics_matrix = calibration_results["master_to_camera"]
        else:
            extrinsics_matrix = calibration_results

        self.save_finalised_bundle(extrinsics_matrix)

    def save_finalised_bundle(self, extrinsics_matrix):
        bundle_dir = os.path.dirname(self.bundle_path)
        bundle_stem = os.path.splitext(os.path.basename(self.bundle_path))[0]
        default_path = os.path.join(bundle_dir, f"{bundle_stem}_finalised_extrinsics.npz")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Finalised Calibration Bundle", default_path, "NumPy Bundle (*.npz)"
        )
        if not file_path:
            return

        np.savez(
            file_path,
            intrinsics_matrix=self.intrinsics_matrix,
            distortion_matrix=self.distortion_matrix,
            extrinsics_matrix=np.asarray(extrinsics_matrix, dtype=np.float64),
        )
        print(f"Finalised calibration bundle saved to {file_path}")

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------
    def process_dropped_path(self, path):
        lower = path.lower()
        if lower.endswith(POINTCLOUD_EXTENSIONS):
            self.set_laz_path(path)
        elif lower.endswith(IMAGE_EXTENSIONS):
            self.set_image_path(path)
        elif lower.endswith(BUNDLE_EXTENSIONS):
            self.set_bundle_path(path)
        elif lower.endswith(POINTS_EXTENSIONS):
            # Image points have X and Y column names; LiDAR points are x, y, z rows
            if is_image_points_csv(path):
                self.set_image_points_path(path)
            else:
                self.set_points_path(path)
        else:
            self.load_status_label.setText(f"Error: Unrecognised file type: {path}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            self.load_status_label.setText("")
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isfile(path):
                    self.process_dropped_path(path)
            event.acceptProposedAction()
