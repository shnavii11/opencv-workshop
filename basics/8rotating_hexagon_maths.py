# Concept: Trigonometry-Based Animation
# A hexagon has 6 vertices evenly spaced around a circle (every 60 degrees).
# Using sin and cos, we can calculate where each vertex is at any angle —
# and by changing that angle over time, the hexagon spins.

import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)   # Open default webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)   # Mirror so it feels like a selfie cam
    h, w, _ = frame.shape

    cx, cy = w // 2, h // 2      # Center of the frame
    radius = 100                 # Distance from center to each vertex

    # angle_offset increases continuously with time → the hexagon rotates
    # Multiply time.time() by 60 to control rotation speed (degrees per second)
    angle_offset = time.time() * 60

    points = []
    for i in range(6):
        # Each vertex is 60° apart around the circle (360° / 6 = 60°)
        angle = np.deg2rad(angle_offset + i * 60)

        # Parametric circle equations: x = cx + r*cos(θ), y = cy + r*sin(θ)
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        points.append((x, y))

    # Connect adjacent vertices; (i+1) % 6 wraps the last vertex back to the first
    for i in range(6):
        cv2.line(frame, points[i], points[(i + 1) % 6], (0, 255, 255), 3)

    cv2.imshow("Rotating Hexagon", frame)

    if cv2.waitKey(1) & 0xFF == 27:   # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
