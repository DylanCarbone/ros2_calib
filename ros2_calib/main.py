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

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def parse_args(argv):
    parser = argparse.ArgumentParser(description="ros2_calib - LiDAR-Camera Calibration Tool")
    parser.add_argument("--laz", help="Path to the point cloud (.laz/.las) to pre-fill on launch")
    parser.add_argument("--image", help="Path to the camera image to pre-fill on launch")
    parser.add_argument(
        "--bundle", help="Path to the calibration bundle (.npz) to pre-fill on launch"
    )
    return parser.parse_known_args(argv)[0]


def main():
    args = parse_args(sys.argv[1:])
    app = QApplication(sys.argv)
    window = MainWindow(laz_path=args.laz, image_path=args.image, bundle_path=args.bundle)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
