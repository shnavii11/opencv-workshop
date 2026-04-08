# Concept: Image & Video I/O
# OpenCV can read images from disk, display them, save them,
# and capture live video from a webcam — this file covers all four.

import cv2

# path constants
from pathlib import Path
root_path = Path(__file__).resolve().parent
repo_dir = root_path.parent
assets_dir = repo_dir / "assets"
input_img = assets_dir / "input.png"
video_loc = root_path / 'workshop_output.avi'
last_frame_loc = root_path / 'captured_frame.jpg'

# --- IMAGE IO ---
img = cv2.imread(input_img)        # Load image from disk into a NumPy array
if img is not None:
    cv2.imshow('Static Image (Press any key)', img)   # Pop up a window
    cv2.imwrite('assets/copy_of_input.png', img)      # Save a copy to disk
    cv2.waitKey(0)                                     # Wait for any key before continuing

# --- VIDEO & WEBCAM IO ---
cap = cv2.VideoCapture(0)  # 0 = default webcam (try 1 or 2 if this doesn't work)

# Set up a VideoWriter so we can save the capture to a file
fourcc = cv2.VideoWriter_fourcc(*'XVID')               # XVID is a common compressed format
out = cv2.VideoWriter(video_loc, fourcc, 20.0, (640, 480))

print("Capturing... Press 'q' to stop and save the last frame.")
while cap.isOpened():
    ret, frame = cap.read()    # ret = True if frame was read successfully
    if not ret: break

    cv2.imshow('Webcam Feed', frame)
    out.write(frame)           # Write each frame to the output video file

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(last_frame_loc, frame)  # Save the final frame as a JPEG
        break

# Always release resources when done — leaving them open can crash your next run
cap.release()
out.release()
cv2.destroyAllWindows()
