# OpenCV Workshop

A hands-on introduction to computer vision for first-year engineering students — no prior experience needed. You'll go from reading a single image to projecting a spinning 3D cube, one concept at a time.

---

## What is OpenCV?

OpenCV (Open Source Computer Vision Library) is the go-to tool for anything vision-related in Python. Think of it as numpy, but for images and cameras. It's used everywhere: from detecting faces on your phone to self-driving cars figuring out where the road is.

This workshop teaches you the foundations — how to read, manipulate, filter, and draw on images — and then applies that math to make things move.

---

## Folder Structure

```
opencv-workshop/
├── assets/              # Input images used by the scripts
├── basics/              # Core concepts, meant to be run in order (1 → 9)
│   ├── constants.py         # Shared file paths (don't touch, just import)
│   ├── 1io.py               # Reading/writing images and webcam capture
│   ├── 2shape_colours.py    # Image dimensions and color spaces
│   ├── 3bitwise_threshhold.py # Bitwise ops, masking, thresholding
│   ├── 4drawing_shapes.py   # Drawing shapes and text on images
│   ├── 5convulation_kernel.py # Kernels, filters, and padding
│   ├── 6contours.py         # Edge detection and contour analysis
│   ├── 7morphology.py       # Erosion, dilation, and morphological ops
│   ├── 8rotating_hexagon_maths.py # Animated hexagon using trigonometry
│   └── 9rotating_cube_maths.py    # 3D cube with rotation matrices
├── projects/            # Fun mini-projects built on the basics
│   ├── fruit_ninja.py
│   ├── invisible_cloak.py
│   └── shield.py
├── requirements.txt     # All Python dependencies
└── README.md
```

---

## Basics — What Each File Covers

| File | Concept | One-liner |
|------|---------|-----------|
| `1io.py` | Image & Video I/O | Load an image, show it, save it, capture from webcam |
| `2shape_colours.py` | Shapes & Color Spaces | Images are just arrays; colors depend on the "language" you use |
| `3bitwise_threshhold.py` | Bitwise Ops & Thresholding | Combine or isolate parts of an image using logic gates |
| `4drawing_shapes.py` | Drawing | Paint rectangles, circles, text, and polygons on any image |
| `5convulation_kernel.py` | Convolution Kernels | Slide a small matrix over the image to sharpen, blur, or emboss it |
| `6contours.py` | Contours & Edges | Find object outlines and describe their shape |
| `7morphology.py` | Morphological Ops | Shrink or grow white regions to clean up binary images |
| `8rotating_hexagon_maths.py` | Trig Animation | Use `sin` and `cos` to spin a hexagon on the webcam feed |
| `9rotating_cube_maths.py` | 3D Projection | Rotate a 3D cube and project it flat onto your screen |

---

## Setup

### 1. Make sure you have Python 3.12

```bash
python --version   # Should say 3.12.x
```

Don't have it? Download from [python.org](https://www.python.org/downloads/).

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs OpenCV, NumPy, MediaPipe, and everything else the projects need.

---

## How to Run

All scripts in `basics/` import from `constants.py`, so **run them from inside the `basics/` folder**:

```bash
cd basics
python 1io.py
```

### Quick reference

| Script | Command | Notes |
|--------|---------|-------|
| `1io.py` | `python 1io.py` | Opens webcam — press `q` to stop |
| `2shape_colours.py` | `python 2shape_colours.py` | Press any key to close windows |
| `3bitwise_threshhold.py` | `python 3bitwise_threshhold.py` | Multiple windows open sequentially |
| `4drawing_shapes.py` | `python 4drawing_shapes.py` | Press any key to close |
| `5convulation_kernel.py` | `python 5convulation_kernel.py` | Press any key to close |
| `6contours.py` | `python 6contours.py` | Press any key to close |
| `7morphology.py` | `python 7morphology.py` | Multiple windows, press any key each time |
| `8rotating_hexagon_maths.py` | `python 8rotating_hexagon_maths.py` | Live webcam — press `ESC` to exit |
| `9rotating_cube_maths.py` | `python 9rotating_cube_maths.py` | Live webcam — press `ESC` to exit |

> **Tip:** If a window seems frozen, click on it and press any key. OpenCV needs focus to register keypresses.

---

## Webcam Note

Files `1io.py`, `8rotating_hexagon_maths.py`, and `9rotating_cube_maths.py` open your webcam. Make sure it's connected and not in use by another app (yes, that means closing your Zoom call first).

---

## Prerequisites

You don't need to know computer vision. You do need to be comfortable with:
- Basic Python (variables, loops, functions)
- A little numpy wouldn't hurt, but it's not required

---

## What's Next?

Once you're through the basics, check out the `projects/` folder:
- **Invisible Cloak** — Harry Potter vibes, achieved with color masking
- **Fruit Ninja** — Slash moving objects detected by the camera
- **Shield** — A reactive visual overlay on your hand

These all build directly on the concepts from `basics/`. Good luck — and yes, the rotating cube is as cool as it sounds.
