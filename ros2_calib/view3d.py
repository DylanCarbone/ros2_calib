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
3D point cloud view for the LiDAR-to-Camera calibration widget.

Points are given in the camera frame (x right, y down, z forward) and drawn by
projecting them in numpy with a depth buffer, so no OpenGL is needed. The view
starts at the real camera's position and field of view, so it first looks like
the camera image, and can then be orbited, panned and zoomed. Each set of points
is a named layer (e.g. the LiDAR cloud); a click without dragging can pick the
nearest visible point of a given layer.
"""

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSplitter, QWidget


class PairedViewsSplitter(QSplitter):
    """Two views side by side or one above the other, whichever shows pictures of aspect ratio
    `aspect` (width / height) larger in the space available."""

    def __init__(self, aspect=16 / 9, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.aspect = aspect

    def resizeEvent(self, event):
        width, height = event.size().width(), event.size().height()
        # Width each picture is drawn at, side by side and stacked
        side_by_side = min(width / 2, height * self.aspect)
        stacked = min(width, height / 2 * self.aspect)
        orientation = Qt.Horizontal if side_by_side >= stacked else Qt.Vertical
        if orientation != self.orientation():
            self.setOrientation(orientation)
            # Equal halves
            self.setSizes([10000, 10000])
        super().resizeEvent(event)


class PointLayer:
    """Points (N x 3, camera frame) and their colours (N x 3 uint8 RGB), each drawn as a square of
    `point_size` px."""

    def __init__(self, points, colors, point_size):
        self.points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        self.colors = np.ascontiguousarray(colors, dtype=np.uint8).reshape(-1, 3)
        self.point_size = max(1, int(point_size))
        self.visible = True
        # Fixed quarter of the points, drawn while the view is being dragged
        count = len(self.points)
        self.preview = np.random.default_rng(0).permutation(count)[: max(1, count // 4)]


class PointCloudView3D(QWidget):
    """Software-rendered 3D view of point layers, with orbit/pan/zoom and point picking."""

    # Press and release of the left button without dragging, at this widget position
    clicked = Signal(float, float)

    BACKGROUND = (35, 35, 35)
    NEAR_CLIP_M = 0.05
    # Mouse movement (px) before a press counts as a drag rather than a click
    DRAG_THRESHOLD_PX = 4
    ORBIT_DEG_PER_PX = 0.3
    ZOOM_PER_WHEEL_STEP = 0.85
    # Full-quality redraw after interaction has paused this long (ms)
    REFINE_DELAY_MS = 150
    # Relative depth difference within which picked points count as on the same surface
    SAME_SURFACE_DEPTH = 0.03

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = {}
        # Callable returning overlay markers drawn above the points, as a list of
        # (points (M x 3), QColor, radius px, style), style "dot", "ring" or "cross"
        self.marker_provider = None
        # Real camera intrinsics, for the starting view
        self.camera_intrinsics = (1000.0, 1000.0, 960.0, 540.0)
        self.camera_image_size = (1920, 1080)
        self.yaw = 0.0
        self.pitch = 0.0
        self.distance = 10.0
        self.pivot = np.array([0.0, 0.0, 10.0])
        self._frame = None
        self._frame_is_preview = False
        self._press_pos = None
        self._last_pos = None
        self._press_button = None
        self._dragging = False
        self._refine_timer = QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.timeout.connect(self._refine)
        self.setMinimumSize(200, 150)
        self.setMouseTracking(False)
        self.setToolTip(
            "Left-drag: orbit · Right-drag: pan · Wheel: zoom · Middle-click: orbit around "
            "the point clicked"
        )

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------
    def set_camera(self, fx, fy, cx, cy, width, height):
        """Real camera intrinsics and image size, which set the starting view."""
        self.camera_intrinsics = (float(fx), float(fy), float(cx), float(cy))
        self.camera_image_size = (int(width), int(height))

    def set_layer(self, name, points, colors, point_size=2):
        """Add or replace a layer, keeping its visibility."""
        visible = self.layers[name].visible if name in self.layers else True
        self.layers[name] = PointLayer(points, colors, point_size)
        self.layers[name].visible = visible
        self.request_render()

    def set_layer_visible(self, name, visible):
        if name in self.layers:
            self.layers[name].visible = bool(visible)
            self.request_render()

    def reset_view(self, pivot_depth=None):
        """Return to the real camera's viewpoint, orbiting around the point straight ahead at
        `pivot_depth` (m; default: median depth of the visible points)."""
        if pivot_depth is None:
            depths = [layer.points[:, 2] for layer in self.layers.values() if len(layer.points) > 0]
            depths = np.concatenate(depths) if depths else np.array([10.0])
            depths = depths[depths > self.NEAR_CLIP_M]
            pivot_depth = float(np.median(depths)) if depths.size else 10.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.distance = pivot_depth
        self.pivot = np.array([0.0, 0.0, pivot_depth])
        self.request_render()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def view_rotation(self):
        """Rotation from the camera frame to the view frame: pitch about x, after yaw about y."""
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        return r_pitch @ r_yaw

    def view_position(self):
        """Viewpoint in the camera frame, `distance` behind the pivot along the view direction."""
        forward = self.view_rotation().T @ np.array([0.0, 0.0, 1.0])
        return self.pivot - self.distance * forward

    def view_intrinsics(self):
        """Focal lengths and principal point of the view (px), the real camera's scaled so its
        image fits the widget."""
        fx, fy, cx, cy = self.camera_intrinsics
        image_w, image_h = self.camera_image_size
        scale = min(self.width() / image_w, self.height() / image_h)
        offset_x = (self.width() - image_w * scale) / 2
        offset_y = (self.height() - image_h * scale) / 2
        return fx * scale, fy * scale, cx * scale + offset_x, cy * scale + offset_y

    def project(self, points):
        """Widget pixel coordinates (N x 2) and view depth (N) of camera-frame points."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        in_view = (points - self.view_position()) @ self.view_rotation().T
        fx, fy, cx, cy = self.view_intrinsics()
        z = in_view[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * in_view[:, 0] / z + cx
            v = fy * in_view[:, 1] / z + cy
        return np.column_stack([u, v]), z

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def request_render(self, preview=False):
        """Redraw now (a quick preview while interacting, refined shortly after)."""
        self._render(preview)
        self.update()
        if preview:
            self._refine_timer.start(self.REFINE_DELAY_MS)

    def _refine(self):
        if self._frame_is_preview and not self._dragging:
            self.request_render()

    @staticmethod
    def _splat(u, v, size, width, height):
        """Pixels covered by points drawn as squares of `size` px around integer pixels (u, v),
        within a width x height area: (flattened pixel index, index of the point) pairs."""
        pixels, owners = [], []
        points = np.arange(len(u))
        lo = -(int(size) // 2)
        for dv in range(lo, lo + int(size)):
            for du in range(lo, lo + int(size)):
                pu, pv = u + du, v + dv
                inside = (pu >= 0) & (pu < width) & (pv >= 0) & (pv < height)
                pixels.append(pv[inside] * width + pu[inside])
                owners.append(points[inside])
        if not pixels:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(pixels), np.concatenate(owners)

    def _layer_pixels(self, layer, preview=False):
        """Pixel indices (flattened), depths and colours of a layer's points drawn in the view,
        each point covering a square of its size. A preview draws a quarter of the points."""
        width, height = self.width(), self.height()
        subset = layer.preview if preview else np.arange(len(layer.points))
        uv, z = self.project(layer.points[subset])
        margin = layer.point_size
        keep = (
            (z > self.NEAR_CLIP_M)
            & (uv[:, 0] > -margin)
            & (uv[:, 0] < width + margin)
            & (uv[:, 1] > -margin)
            & (uv[:, 1] < height + margin)
        )
        subset, uv, z = subset[keep], uv[keep], z[keep]
        pixels, owners = self._splat(
            np.floor(uv[:, 0]).astype(np.int64),
            np.floor(uv[:, 1]).astype(np.int64),
            layer.point_size,
            width,
            height,
        )
        return pixels, z[owners], layer.colors[subset[owners]]

    def _render(self, preview=False):
        width, height = self.width(), self.height()
        if width < 2 or height < 2:
            self._frame = None
            return
        depth_buffer = np.full(width * height, np.inf, dtype=np.float64)
        color_buffer = np.empty((width * height, 3), dtype=np.uint8)
        color_buffer[:] = self.BACKGROUND
        parts = [
            self._layer_pixels(layer, preview)
            for layer in self.layers.values()
            if layer.visible and len(layer.points) > 0
        ]
        if parts:
            pixels = np.concatenate([p[0] for p in parts])
            depths = np.concatenate([p[1] for p in parts])
            colors = np.concatenate([p[2] for p in parts])
            # Nearest point at each pixel wins
            np.minimum.at(depth_buffer, pixels, depths)
            front = depths <= depth_buffer[pixels]
            color_buffer[pixels[front]] = colors[front]
        self._frame = np.ascontiguousarray(color_buffer.reshape(height, width, 3))
        self._frame_is_preview = preview

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._frame is None or self._frame.shape[:2] != (self.height(), self.width()):
            self._render()
        if self._frame is not None:
            height, width = self._frame.shape[:2]
            image = QImage(self._frame.data, width, height, 3 * width, QImage.Format_RGB888)
            painter.drawImage(0, 0, image)
        self._draw_markers(painter)
        painter.end()

    def _draw_markers(self, painter):
        if self.marker_provider is None:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        for marker in self.marker_provider():
            # (points, colour, radius, style) and optionally a text label per point
            points, color, radius, style = marker[:4]
            labels = marker[4] if len(marker) > 4 else None
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            if len(points) == 0:
                continue
            uv, z = self.project(points)
            outline = QPen(QColor("white"), 1.5)
            for k, ((u, v), depth) in enumerate(zip(uv, z)):
                if depth <= self.NEAR_CLIP_M:
                    continue
                center = QPointF(u, v)
                if labels is not None:
                    # Label at the upper right, with a dark shadow to read on any background
                    text_pos = QPointF(u + radius + 2, v - radius - 2)
                    painter.setPen(QPen(QColor("black")))
                    painter.drawText(text_pos + QPointF(1, 1), labels[k])
                    painter.setPen(QPen(color))
                    painter.drawText(text_pos, labels[k])
                if style == "cross":
                    painter.setPen(QPen(color, 2.5))
                    painter.drawLine(QPointF(u - radius, v), QPointF(u + radius, v))
                    painter.drawLine(QPointF(u, v - radius), QPointF(u, v + radius))
                elif style == "ring":
                    painter.setPen(QPen(color, 2.5))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(center, radius, radius)
                else:
                    painter.setPen(outline)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(center, radius, radius)

    def resizeEvent(self, event):
        self._frame = None
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------
    def pick(self, x, y, layer_name, radius_px=8):
        """Index of the `layer_name` point seen at widget position (x, y), or else at the
        nearest pixel within `radius_px` where one is seen; None when there is none. Points are
        drawn as in the view, so a point hidden behind another (or under its drawn square) can't
        be picked; only that layer's points count, so other layers in front don't block it."""
        layer = self.layers.get(layer_name)
        if layer is None or not layer.visible or len(layer.points) == 0:
            return None
        uv, z = self.project(layer.points)
        reach = int(radius_px) + layer.point_size
        candidates = np.where(
            (z > self.NEAR_CLIP_M)
            & (np.abs(uv[:, 0] - x) <= reach)
            & (np.abs(uv[:, 1] - y) <= reach)
        )[0]
        if candidates.size == 0:
            return None
        # Draws the candidates into a small depth buffer centred on the clicked pixel
        size = 2 * reach + 1
        origin_u, origin_v = int(np.floor(x)) - reach, int(np.floor(y)) - reach
        pixels, owners = self._splat(
            np.floor(uv[candidates, 0]).astype(np.int64) - origin_u,
            np.floor(uv[candidates, 1]).astype(np.int64) - origin_v,
            layer.point_size,
            size,
            size,
        )
        depth = np.full(size * size, np.inf)
        owner_depth = z[candidates][owners]
        np.minimum.at(depth, pixels, owner_depth)
        front = owner_depth <= depth[pixels]
        winner = np.full(size * size, -1, dtype=np.int64)
        winner[pixels[front]] = owners[front]
        covered = np.where(winner >= 0)[0]
        if covered.size == 0:
            return None
        # The point seen at the clicked pixel, or at the nearest pixel where the layer is seen
        dist2 = (covered % size - reach) ** 2 + (covered // size - reach) ** 2
        best = int(np.argmin(dist2))
        if dist2[best] > radius_px**2:
            return None
        seen = winner[covered[best]]
        # Its drawn square can hide neighbours on the same surface whose centres are nearer the
        # click, so the point picked is the one nearest the click among those at about its
        # depth (points hidden behind it are deeper, so they never are)
        candidate_z = z[candidates]
        same_surface = (
            np.abs(candidate_z - candidate_z[seen]) <= self.SAME_SURFACE_DEPTH * candidate_z[seen]
        )
        centre_dist2 = (uv[candidates, 0] - x) ** 2 + (uv[candidates, 1] - y) ** 2
        centre_dist2[~same_surface] = np.inf
        return int(candidates[int(np.argmin(centre_dist2))])

    def pick_any(self, x, y, radius_px=8):
        """Camera-frame position of the front-most visible point of any layer nearest to widget
        position (x, y), or None."""
        best = None
        for name, layer in self.layers.items():
            index = self.pick(x, y, name, radius_px)
            if index is None:
                continue
            point = layer.points[index].astype(np.float64)
            _, depth = self.project(point)
            if best is None or depth[0] < best[1]:
                best = (point, depth[0])
        return None if best is None else best[0]

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        self._press_pos = event.position()
        self._last_pos = event.position()
        self._press_button = event.button()
        self._dragging = False

    def mouseMoveEvent(self, event):
        if self._press_pos is None:
            return
        pos = event.position()
        if not self._dragging:
            moved = pos - self._press_pos
            if abs(moved.x()) + abs(moved.y()) < self.DRAG_THRESHOLD_PX:
                return
            self._dragging = True
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        if self._press_button == Qt.LeftButton:
            self.orbit(dx, dy)
        elif self._press_button in (Qt.RightButton, Qt.MiddleButton):
            self.pan(dx, dy)

    def mouseReleaseEvent(self, event):
        if self._press_pos is not None and not self._dragging:
            pos = event.position()
            if self._press_button == Qt.LeftButton:
                self.clicked.emit(pos.x(), pos.y())
            elif self._press_button == Qt.MiddleButton:
                self.center_on(pos.x(), pos.y())
        was_dragging = self._dragging
        self._press_pos = None
        self._dragging = False
        if was_dragging:
            self.request_render()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self.zoom(self.ZOOM_PER_WHEEL_STEP**steps)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def orbit(self, dx, dy):
        """Turn the view around the pivot, as if dragging the scene by (dx, dy) px."""
        self.yaw -= np.radians(dx * self.ORBIT_DEG_PER_PX)
        self.pitch = float(
            np.clip(self.pitch + np.radians(dy * self.ORBIT_DEG_PER_PX), -1.55, 1.55)
        )
        self.request_render(preview=True)

    def pan(self, dx, dy):
        """Slide the view so the scene at the pivot's depth follows the mouse by (dx, dy) px."""
        fx, fy, _, _ = self.view_intrinsics()
        rotation_t = self.view_rotation().T
        self.pivot = self.pivot - rotation_t @ np.array(
            [dx * self.distance / fx, dy * self.distance / fy, 0.0]
        )
        self.request_render(preview=True)

    def zoom(self, factor):
        """Move the viewpoint towards (factor < 1) or away from the pivot."""
        self.distance = float(np.clip(self.distance * factor, 0.05, 1e5))
        self.request_render(preview=True)

    def center_on(self, x, y):
        """Orbit around the point under widget position (x, y) from now on, keeping the
        viewpoint where it is and turning to face that point."""
        target = self.pick_any(x, y, radius_px=12)
        if target is None:
            return
        position = self.view_position()
        offset = target - position
        distance = float(np.linalg.norm(offset))
        if distance < 1e-6:
            return
        forward = offset / distance
        # Inverse of view_position's forward = R^T e_z = (-sin yaw cos pitch, sin pitch,
        # cos yaw cos pitch), with R = R_pitch @ R_yaw
        self.pitch = float(np.clip(np.arcsin(forward[1]), -1.55, 1.55))
        self.yaw = float(np.arctan2(-forward[0], forward[2]))
        self.distance = distance
        self.pivot = target
        self.request_render()
