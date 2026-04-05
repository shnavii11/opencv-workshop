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
img = cv2.imread(input_img)
if img is not None:
    cv2.imshow('Static Image (Press any key)', img)
    cv2.imwrite('assets/copy_of_input.png', img)
    cv2.waitKey(0)

# --- VIDEO & WEBCAM IO ---
cap = cv2.VideoCapture(0) # '0' is default webcam
# Define codec and create VideoWriter object to save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_loc, fourcc, 20.0, (640, 480))

print("Capturing... Press 'q' to stop and save the last frame.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow('Webcam Feed', frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(last_frame_loc, frame) # Capture single frame
        break

cap.release()
out.release()
cv2.destroyAllWindows()