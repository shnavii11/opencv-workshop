# Concept: 3D Rotation & Orthographic Projection
# A 3D cube has 8 vertices in (x, y, z) space. To draw it on a 2D screen,
# we rotate the vertices using rotation matrices, then "flatten" them by ignoring Z.
# This is called orthographic projection — no perspective, but it shows depth clearly.

import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

# 8 corners of a unit cube centered at the origin, in 3D (x, y, z)
vertices = np.array([
    [-1, -1, -1],   # 0: back-bottom-left
    [ 1, -1, -1],   # 1: back-bottom-right
    [ 1,  1, -1],   # 2: back-top-right
    [-1,  1, -1],   # 3: back-top-left
    [-1, -1,  1],   # 4: front-bottom-left
    [ 1, -1,  1],   # 5: front-bottom-right
    [ 1,  1,  1],   # 6: front-top-right
    [-1,  1,  1]    # 7: front-top-left
])

# 12 edges: each is a pair of vertex indices to connect with a line
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # Back face
    (4, 5), (5, 6), (6, 7), (7, 4),   # Front face
    (0, 4), (1, 5), (2, 6), (3, 7)    # Connecting edges (depth)
]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    cx, cy = w // 2, h // 2   # Center of screen
    scale = 100               # Multiplier to convert unit-cube coords to pixels

    angle_time = time.time() * 1.5    # Y-axis angle changes over time → continuous spin
    angle_45 = math.pi / 4            # Fixed 45° tilt on X-axis so we see top and front

    # Rotation matrix around X-axis (tilts cube up/down)
    # This is the standard 3D rotation matrix for X: https://en.wikipedia.org/wiki/Rotation_matrix
    rot_x = np.array([
        [1,               0,                0],
        [0, math.cos(angle_45), -math.sin(angle_45)],
        [0, math.sin(angle_45),  math.cos(angle_45)]
    ])

    # Rotation matrix around Y-axis (spins cube left/right)
    rot_y = np.array([
        [ math.cos(angle_time), 0, math.sin(angle_time)],
        [0,                     1, 0                   ],
        [-math.sin(angle_time), 0, math.cos(angle_time)]
    ])

    projected_points = []
    for v in vertices:
        # Apply Y rotation first (spin), then X rotation (tilt)
        rotated_v = np.dot(rot_y, v)
        rotated_v = np.dot(rot_x, rotated_v)

        # Orthographic projection: just drop Z and map (x, y) to screen coordinates
        x = int(cx + rotated_v[0] * scale)
        y = int(cy + rotated_v[1] * scale)
        projected_points.append((x, y))

    # Draw each edge by connecting its two projected 2D points
    for edge in edges:
        pt1 = projected_points[edge[0]]
        pt2 = projected_points[edge[1]]
        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

    cv2.imshow("Rotating 3D Cube", frame)

    if cv2.waitKey(1) & 0xFF == 27:   # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
